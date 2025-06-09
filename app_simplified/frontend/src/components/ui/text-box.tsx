import * as React from "react"
import { cn } from "../../lib/utils"

interface TextBoxProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
  helperText?: string
  maxLength?: number
  showCount?: boolean
  resize?: 'none' | 'vertical' | 'horizontal' | 'both'
}

const TextBox = React.forwardRef<HTMLTextAreaElement, TextBoxProps>(
  ({ 
    className, 
    label, 
    error, 
    helperText, 
    maxLength, 
    showCount = false,
    resize = 'vertical',
    ...props 
  }, ref) => {
    const [charCount, setCharCount] = React.useState(props.value?.toString().length || 0)

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setCharCount(e.target.value.length)
      props.onChange?.(e)
    }

    const resizeStyles = {
      none: 'resize-none',
      vertical: 'resize-y',
      horizontal: 'resize-x',
      both: 'resize'
    }

    return (
      <div className="space-y-2">
        {label && (
          <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
            {label}
          </label>
        )}
        <div className="relative">
          <textarea
            ref={ref}
            className={cn(
              "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
              resizeStyles[resize],
              error && "border-red-500 focus-visible:ring-red-500",
              className
            )}
            maxLength={maxLength}
            onChange={handleChange}
            {...props}
          />
          {showCount && maxLength && (
            <div className="absolute bottom-2 right-2 text-xs text-muted-foreground">
              {charCount}/{maxLength}
            </div>
          )}
        </div>
        {(error || helperText) && (
          <p className={cn(
            "text-sm",
            error ? "text-red-500" : "text-muted-foreground"
          )}>
            {error || helperText}
          </p>
        )}
      </div>
    )
  }
)
TextBox.displayName = "TextBox"

export { TextBox } 