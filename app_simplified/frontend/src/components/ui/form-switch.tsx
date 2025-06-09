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
                    "relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border transition-colors duration-200 ease-in-out focus:outline-none",
                    checked 
                        ? "bg-blue-100 border-blue-200" 
                        : "bg-gray-100 border-gray-200",
                    disabled && "cursor-not-allowed opacity-50"
                )}
            >
                <span
                    className={cn(
                        "pointer-events-none inline-block h-4 w-4 transform rounded-full transition duration-200 ease-in-out absolute top-1",
                        checked 
                            ? "translate-x-6 bg-blue-500" 
                            : "translate-x-1 bg-gray-400"
                    )}
                />
            </button>
            <div className="space-y-1">
                <Label className="text-sm font-medium text-gray-700">{label}</Label>
                {description && (
                    <p className="text-sm text-gray-500">{description}</p>
                )}
            </div>
        </div>
    );
}; 