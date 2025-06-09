import * as React from "react"
import { cn } from "../../lib/utils"

interface SwitchProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'type'> {
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
}

const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(
  ({ className, checked = false, onCheckedChange, ...props }, ref) => {
    const handleClick = React.useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault()
      onCheckedChange?.(!checked)
    }, [checked, onCheckedChange])

    return (
      <button
        ref={ref}
        type="button"
        role="switch"
        aria-checked={checked}
        data-state={checked ? "checked" : "unchecked"}
        className={cn(
          "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer",
          checked ? "bg-primary" : "bg-gray-300 dark:bg-gray-600",
          className
        )}
        onClick={handleClick}
        style={{ WebkitTapHighlightColor: "transparent" }}
        {...props}
      >
        <span
          className={cn(
            "pointer-events-none absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform duration-200 ease-in-out",
            checked ? "translate-x-5" : "translate-x-0"
          )}
          style={{ WebkitTapHighlightColor: "transparent" }}
        />
      </button>
    )
  }
)
Switch.displayName = "Switch"

export { Switch } 