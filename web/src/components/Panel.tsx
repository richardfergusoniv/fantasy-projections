import type { ReactNode } from "react";

export function Panel({
  title,
  actions,
  children,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel" aria-label={title}>
      <header className="panel-header">
        <h2>{title}</h2>
        {actions ? <div className="panel-tools">{actions}</div> : null}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
