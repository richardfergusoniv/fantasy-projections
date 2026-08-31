/** Shared column-sort helpers for Fantasy Tools tables. */
(function (global) {
  function compareValues(av, bv, dir) {
    const mult = dir === "desc" ? -1 : 1;
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string" || typeof bv === "string") {
      return String(av).localeCompare(String(bv), undefined, { sensitivity: "base" }) * mult;
    }
    const an = Number(av);
    const bn = Number(bv);
    if (Number.isNaN(an) && Number.isNaN(bn)) return 0;
    if (Number.isNaN(an)) return 1;
    if (Number.isNaN(bn)) return -1;
    return (an - bn) * mult;
  }

  function sortRows(rows, { key, dir, getValue }) {
    const getter = getValue || ((row) => row[key]);
    return [...rows].sort((a, b) => compareValues(getter(a), getter(b), dir));
  }

  function toggleSort(state, key, { keyProp = "key", dirProp = "dir", defaultDir = "asc" } = {}) {
    if (state[keyProp] === key) {
      state[dirProp] = state[dirProp] === "asc" ? "desc" : "asc";
    } else {
      state[keyProp] = key;
      state[dirProp] = defaultDir;
    }
    return state;
  }

  function thAttrs({ key, label, className = "", sortKey, sortDir, defaultDir = "asc" }) {
    const classes = ["sortable", className].filter(Boolean);
    if (sortKey === key) {
      classes.push(sortDir === "desc" ? "sorted-desc" : "sorted-asc");
    }
    return `<th class="${classes.join(" ").trim()}" data-sort="${key}" data-default-dir="${defaultDir}" role="columnheader" tabindex="0" title="Sort by ${label}">${label}</th>`;
  }

  function bindHeader(headEl, onToggle) {
    if (!headEl || headEl.dataset.sortBound === "1") return;
    headEl.dataset.sortBound = "1";
    headEl.addEventListener("click", (e) => {
      const th = e.target.closest("th.sortable");
      if (!th) return;
      onToggle(th.dataset.sort, th.dataset.defaultDir || "asc");
    });
    headEl.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      const th = e.target.closest("th.sortable");
      if (!th) return;
      e.preventDefault();
      onToggle(th.dataset.sort, th.dataset.defaultDir || "asc");
    });
  }

  function markStaticHeaders(root, sortKey, sortDir) {
    root.querySelectorAll("th.sortable").forEach((th) => {
      th.classList.toggle("sorted-asc", th.dataset.sort === sortKey && sortDir === "asc");
      th.classList.toggle("sorted-desc", th.dataset.sort === sortKey && sortDir === "desc");
    });
  }

  global.FantasySort = {
    compareValues,
    sortRows,
    toggleSort,
    thAttrs,
    bindHeader,
    markStaticHeaders,
  };
})(window);
