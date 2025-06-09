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
          "relative inline-flex h-6 w-11 items-center rounded-full transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer border-2",
          checked 
            ? "bg-green-600 border-green-700" 
            : "bg-gray-300 border-gray-400",
          className
        )}
        onClick={handleClick}
        style={{ WebkitTapHighlightColor: "transparent" }}
        {...props}
      >
        <span
          className={cn(
            "pointer-events-none absolute left-0.5 top-0.5 h-4 w-4 rounded-full transition-all duration-300 ease-in-out",
            checked 
              ? "translate-x-5 bg-white" 
              : "translate-x-0 bg-white",
            "shadow-md"
          )}
          style={{ WebkitTapHighlightColor: "transparent" }}
        />
      </button>
    )
  }
)
Switch.displayName = "Switch"

export { Switch } 