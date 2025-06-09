import React from 'react';
import { cn } from '../../lib/utils';
import { Label } from './label';

interface FormSwitchProps {
    label: string;
    description?: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    disabled?: boolean;
    className?: string;
}

export const FormSwitch: React.FC<FormSwitchProps> = ({
    label,
    description,
    checked,
    onChange,
    disabled,
    className
}) => {
    return (
        <div className={cn("flex items-start space-x-3", className)}>
            <button
                type="button"
                role="switch"
                aria-checked={checked}
                disabled={disabled}
                onClick={() => onChange(!checked)}
                className={cn(
                    "relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
                    checked ? "bg-primary" : "bg-input",
                    disabled && "cursor-not-allowed opacity-50"
                )}
            >
                <span
                    className={cn(
                        "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
                        checked ? "translate-x-5" : "translate-x-0"
                    )}
                />
            </button>
            <div className="space-y-1">
                <Label className="text-sm font-medium">{label}</Label>
                {description && (
                    <p className="text-sm text-muted-foreground">{description}</p>
                )}
            </div>
        </div>
    );
}; 