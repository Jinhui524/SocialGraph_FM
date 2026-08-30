import { useRef, type PointerEvent as ReactPointerEvent } from "react";

interface WorkspaceResizeHandleProps {
  readonly axis: "vertical" | "horizontal";
  readonly label: string;
  readonly value: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly onDelta: (delta: number) => void;
  readonly onReset: () => void;
  readonly className?: string;
}

export function WorkspaceResizeHandle({
  axis,
  label,
  value,
  minimum,
  maximum,
  onDelta,
  onReset,
  className = "",
}: WorkspaceResizeHandleProps) {
  const lastPointRef = useRef<number | null>(null);
  const coordinate = (event: ReactPointerEvent) => axis === "vertical" ? event.clientX : event.clientY;
  const finishResize = (target: EventTarget & HTMLDivElement, pointerId?: number) => {
    if (pointerId !== undefined && target.hasPointerCapture(pointerId)) {
      target.releasePointerCapture(pointerId);
    }
    lastPointRef.current = null;
    target.classList.remove("is-active");
  };

  return (
    <div
      className={`workspace-resizer is-${axis} ${className}`}
      role="separator"
      aria-label={label}
      aria-orientation={axis}
      aria-valuemin={minimum}
      aria-valuemax={maximum}
      aria-valuenow={Math.round(value)}
      tabIndex={0}
      onDoubleClick={onReset}
      onPointerDown={(event) => {
        lastPointRef.current = coordinate(event);
        event.currentTarget.setPointerCapture(event.pointerId);
        event.currentTarget.classList.add("is-active");
      }}
      onPointerMove={(event) => {
        if (!event.currentTarget.hasPointerCapture(event.pointerId) || lastPointRef.current === null) return;
        const next = coordinate(event);
        onDelta(next - lastPointRef.current);
        lastPointRef.current = next;
      }}
      onPointerUp={(event) => {
        finishResize(event.currentTarget, event.pointerId);
      }}
      onPointerCancel={(event) => {
        finishResize(event.currentTarget, event.pointerId);
      }}
      onLostPointerCapture={(event) => finishResize(event.currentTarget)}
      onKeyDown={(event) => {
        const decrement = axis === "vertical" ? event.key === "ArrowLeft" : event.key === "ArrowUp";
        const increment = axis === "vertical" ? event.key === "ArrowRight" : event.key === "ArrowDown";
        if (!decrement && !increment) return;
        event.preventDefault();
        const step = event.shiftKey ? 40 : 10;
        onDelta(increment ? step : -step);
      }}
    >
      <span aria-hidden="true" />
    </div>
  );
}
