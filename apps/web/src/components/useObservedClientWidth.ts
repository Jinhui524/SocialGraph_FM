import { useCallback, useEffect, useRef, useState, type RefCallback } from "react";

/** Observes the actual client width of a rendered layout owner. */
export function useObservedClientWidth<T extends HTMLElement>(): readonly [RefCallback<T>, number] {
  const cleanupRef = useRef<(() => void) | null>(null);
  const [clientWidth, setClientWidth] = useState(0);

  const attach = useCallback<RefCallback<T>>((node) => {
    cleanupRef.current?.();
    cleanupRef.current = null;
    if (!node) return;

    const read = () => setClientWidth(Math.max(0, Math.round(node.clientWidth)));
    read();
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(read);
      observer.observe(node);
      cleanupRef.current = () => observer.disconnect();
      return;
    }

    window.addEventListener("resize", read);
    cleanupRef.current = () => window.removeEventListener("resize", read);
  }, []);

  useEffect(() => () => cleanupRef.current?.(), []);
  return [attach, clientWidth] as const;
}
