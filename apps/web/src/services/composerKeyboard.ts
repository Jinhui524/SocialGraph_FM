export interface ComposerKeyEvent {
  readonly key: string;
  readonly shiftKey: boolean;
  readonly isComposing: boolean;
}

export function shouldSubmitComposerKey(event: ComposerKeyEvent): boolean {
  return event.key === "Enter" && !event.shiftKey && !event.isComposing;
}
