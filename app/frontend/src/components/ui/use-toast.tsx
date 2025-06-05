import * as React from "react";

type ToastProps = {
  title?: string;
  description?: string;
  variant?: "default" | "destructive";
};

type ToastActionElement = React.ReactElement;

export const useToast = () => {
  const toast = ({ title, description, variant = "default" }: ToastProps) => {
    // Simple implementation using alert for now
    const message = [title, description].filter(Boolean).join(": ");
    if (variant === "destructive") {
      console.error(message);
      alert("Error: " + message);
    } else {
      console.log(message);
      alert(message);
    }
  };

  return { toast };
};
