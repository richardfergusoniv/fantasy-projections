/**
 * Resolve one sealed release namespace for the whole page load.
 *
 * 1. Fetch active_release_<season>.json with cache revalidation.
 * 2. If the pointer is missing (404/network absence), bootstrap legacy files.
 * 3. If the pointer exists but is malformed, or the manifest/files are invalid,
 *    fail visibly. Never fall back to legacy in that case.
 * 4. Freeze namespace at first successful pointer read so a promotion during
 *    this page load cannot mix two bundles.
 */
(function (global) {
  const POINTER_SCHEMA = "active_release_pointer_v1";
  const MANIFEST_SCHEMA = "release_bundle_manifest_v1";

  class ReleaseLoadError extends Error {
    constructor(message, extra) {
      super(message);
      this.name = "ReleaseLoadError";
      this.fatal = true;
      Object.assign(this, extra || {});
    }
  }

  async function sha256Hex(buffer) {
    const hash = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  function pointerUrl(dataRoot, season) {
    return `${dataRoot.replace(/\/$/, "")}/active_release_${season}.json`;
  }

  function legacyUrls(dataRoot, season) {
    const root = dataRoot.replace(/\/$/, "");
    return {
      players: `${root}/players_${season}.json`,
      team_stats: `${root}/team_stats_${season}.json`,
      comparison: `${root}/comparison_${season}.json`,
      deep_band_accuracy: `${root}/deep_band_accuracy.json`,
    };
  }

  function validatePointer(payload, season) {
    if (!payload || typeof payload !== "object") {
      throw new ReleaseLoadError("Active pointer is not a JSON object");
    }
    if (payload.schema_version !== POINTER_SCHEMA) {
      throw new ReleaseLoadError("Active pointer schema is unsupported");
    }
    if (payload.status !== "active") {
      throw new ReleaseLoadError("Active pointer status is not active");
    }
    if (Number(payload.season) !== Number(season)) {
      throw new ReleaseLoadError("Active pointer season does not match this page");
    }
    if (!payload.namespace || !payload.release_id || !payload.manifest_sha256) {
      throw new ReleaseLoadError("Active pointer is missing identity fields");
    }
    if (!payload.manifest_path && !payload.public_base) {
      throw new ReleaseLoadError("Active pointer cannot resolve the sealed manifest");
    }
    return payload;
  }

  function validateManifest(payload, pointer) {
    if (!payload || payload.schema_version !== MANIFEST_SCHEMA) {
      throw new ReleaseLoadError("Sealed manifest schema is unsupported");
    }
    if (payload.status) {
      throw new ReleaseLoadError("Sealed manifest must not carry mutable status");
    }
    if (!payload.bundle || payload.bundle.namespace !== pointer.namespace) {
      throw new ReleaseLoadError("Sealed manifest namespace does not match the pointer");
    }
    if (Number(payload.bundle.season) !== Number(pointer.season)) {
      throw new ReleaseLoadError("Sealed manifest season does not match the pointer");
    }
    if (!Array.isArray(payload.artifacts) || !payload.artifacts.length) {
      throw new ReleaseLoadError("Sealed manifest has no artifacts");
    }
    return payload;
  }

  function resolveUrls(dataRoot, pointer, manifest) {
    const prefix = dataRoot.replace(/\/$/, "");
    const namespace = pointer.namespace;
    const urls = {
      manifest: `${prefix}/releases/${namespace}/release_bundle_manifest.json`,
    };
    for (const entry of manifest.artifacts) {
      if (!entry.browser_consumed) continue;
      urls[entry.role] = `${prefix}/releases/${namespace}/${entry.path}`;
    }
    return urls;
  }

  async function loadContext(options) {
    const season = options.season;
    const dataRoot = options.dataRoot || "data";
    const fetchImpl = options.fetchImpl || fetch.bind(global);
    const pointerHref = pointerUrl(dataRoot, season);
    let pointerRes;
    try {
      pointerRes = await fetchImpl(pointerHref, { cache: "no-cache" });
    } catch (err) {
      // Network-level absence of the pointer file is a missing pointer.
      return {
        mode: "legacy",
        namespace: null,
        urls: legacyUrls(dataRoot, season),
        pointer: null,
        manifest: null,
        reason: "pointer_unreachable",
      };
    }
    if (pointerRes.status === 404) {
      return {
        mode: "legacy",
        namespace: null,
        urls: legacyUrls(dataRoot, season),
        pointer: null,
        manifest: null,
        reason: "pointer_missing",
      };
    }
    if (!pointerRes.ok) {
      throw new ReleaseLoadError(
        `Active pointer failed to load (${pointerRes.status})`,
        { status: pointerRes.status }
      );
    }
    let pointer;
    try {
      pointer = validatePointer(await pointerRes.json(), season);
    } catch (err) {
      if (err instanceof ReleaseLoadError) throw err;
      throw new ReleaseLoadError("Active pointer is malformed JSON");
    }

    const frozen = {
      namespace: pointer.namespace,
      release_id: pointer.release_id,
      manifest_sha256: String(pointer.manifest_sha256).toLowerCase(),
      season: pointer.season,
    };

    const manifestHref = `${dataRoot.replace(/\/$/, "")}/releases/${frozen.namespace}/release_bundle_manifest.json`;
    const manifestRes = await fetchImpl(manifestHref, { cache: "no-cache" });
    if (!manifestRes.ok) {
      throw new ReleaseLoadError(
        `Sealed manifest missing for namespace ${frozen.namespace} (${manifestRes.status})`
      );
    }
    const manifestBuffer = await manifestRes.arrayBuffer();
    const digest = await sha256Hex(manifestBuffer);
    if (digest !== frozen.manifest_sha256) {
      throw new ReleaseLoadError("Sealed manifest hash does not match the active pointer");
    }
    let manifest;
    try {
      manifest = validateManifest(JSON.parse(new TextDecoder().decode(manifestBuffer)), pointer);
    } catch (err) {
      if (err instanceof ReleaseLoadError) throw err;
      throw new ReleaseLoadError("Sealed manifest is malformed JSON");
    }

    return {
      mode: "namespaced",
      namespace: frozen.namespace,
      releaseId: frozen.release_id,
      pointer,
      manifest,
      urls: resolveUrls(dataRoot, pointer, manifest),
      frozen,
    };
  }

  async function loadJson(context, role, fetchImpl) {
    const impl = fetchImpl || fetch.bind(global);
    const url = context.urls && context.urls[role];
    if (!url) {
      throw new ReleaseLoadError(`Release is missing browser artifact role ${role}`);
    }
    const res = await impl(url, { cache: "no-cache" });
    if (!res.ok) {
      if (context.mode === "legacy") {
        throw new ReleaseLoadError(`Failed to load ${url} (${res.status})`);
      }
      throw new ReleaseLoadError(
        `Namespaced file missing for ${role} in ${context.namespace} (${res.status})`
      );
    }
    return res.json();
  }

  function fail(target, error) {
    const message = error && error.message ? error.message : String(error);
    const html = `<div class="empty-state release-load-error" role="alert">${escapeHtml(message)}</div>`;
    if (!target) {
      document.body.insertAdjacentHTML("afterbegin", html);
      return;
    }
    if (typeof target === "string") {
      const el = document.querySelector(target);
      if (el) {
        el.innerHTML = html;
        return;
      }
    }
    if (target && target.innerHTML !== undefined) {
      target.innerHTML = html;
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  global.FantasyRelease = {
    ReleaseLoadError,
    loadContext,
    loadJson,
    fail,
    pointerUrl,
    legacyUrls,
    validatePointer,
    validateManifest,
    resolveUrls,
  };
})(typeof window !== "undefined" ? window : globalThis);
