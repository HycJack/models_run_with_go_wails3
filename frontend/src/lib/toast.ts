import { useSyncExternalStore } from "react";

type Toast = { id: number; message: string; error?: boolean };
let toasts: Toast[] = [];
let listeners: (() => void)[] = [];
let idc = 0;

function emit() {
  listeners.forEach((l) => l());
}

export function toast(message: string, error = false) {
  const id = ++idc;
  toasts = [...toasts, { id, message, error }];
  emit();
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    emit();
  }, 4000);
}

function subscribe(l: () => void) {
  listeners.push(l);
  return () => {
    listeners = listeners.filter((x) => x !== l);
  };
}

export function useToasts() {
  return useSyncExternalStore(subscribe, () => toasts, () => toasts);
}