import { CANONICAL_PRODUCTION_ORIGIN } from "../pwa/canonical";

/** Tiny provenance chip so a stuck PWA shell is obvious. */
export function AppBuildStamp({ className = "muted" }: { className?: string }) {
  return (
    <p className={className} data-testid="app-build-stamp">
      App build {__APP_BUILD_ID__} · {CANONICAL_PRODUCTION_ORIGIN.replace(/^https:\/\//, "")}
    </p>
  );
}
