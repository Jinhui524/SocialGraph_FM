import type { KeyboardEvent, ReactNode } from "react";
import { X } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";

interface WorkspaceDrawerProps {
  readonly title: string;
  readonly description?: string;
  readonly children: ReactNode;
  readonly onClose: () => void;
  readonly wide?: boolean;
}

function isVisibleFocusableElement(element: HTMLElement, panel: HTMLElement): boolean {
  if (element.tabIndex < 0 || element.matches(":disabled")) return false;
  const closedDetails = element.closest("details:not([open])");
  if (closedDetails && !(element.tagName === "SUMMARY" && element.parentElement === closedDetails)) return false;

  for (let current: HTMLElement | null = element; current && current !== panel.parentElement; current = current.parentElement) {
    if (current.hidden || current.getAttribute("aria-hidden") === "true") return false;
    const style = window.getComputedStyle(current);
    if (style.display === "none" || style.visibility === "hidden") return false;
  }
  return true;
}

export function WorkspaceDrawer({ title, description, children, onClose, wide = false }: WorkspaceDrawerProps) {
  const panelRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();

    return () => {
      if (opener?.isConnected) opener.focus();
    };
  }, []);

  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;

    const panel = panelRef.current;
    const focusable = Array.from(panel?.querySelectorAll<HTMLElement>(
      'a[href], button, input:not([type="hidden"]), select, textarea, summary, [contenteditable="true"], [tabindex]',
    ) ?? []).filter((element) => panel !== null && isVisibleFocusableElement(element, panel));
    if (!focusable.length) {
      event.preventDefault();
      return;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div className="workspace-panel-layer" role="presentation" onMouseDown={(event) => {
      if (event.currentTarget === event.target) onClose();
    }}>
      <section
        className={`workspace-panel ${wide ? "is-wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="workspace-panel-title"
        ref={panelRef}
        onKeyDown={handleKeyDown}
      >
        <header className="workspace-panel__header">
          <div>
            <h2 id="workspace-panel-title">{title}</h2>
            {description ? <p>{description}</p> : null}
          </div>
          <button ref={closeButtonRef} type="button" className="icon-button" onClick={onClose} aria-label={`关闭${title}`}>
            <X size={19} />
          </button>
        </header>
        <div className="workspace-panel__body">{children}</div>
      </section>
    </div>
  );
}
