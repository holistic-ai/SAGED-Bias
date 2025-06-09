import React from 'react';
import { cn } from '../../lib/utils';
import { Label } from './label.tsx';
import { Input } from './input.tsx';

interface FormFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string;
    error?: string;
    description?: string;
}

export const FormField: React.FC<FormFieldProps> = ({
    label,
    error,
    description,
    className,
    id,
    ...props
}) => {
    const inputId = id || `field-${label.toLowerCase().replace(/\s+/g, '-')}`;

    return (
        <div className="space-y-2">
            <Label htmlFor={inputId} className="text-sm font-medium">
                {label}
            </Label>
            <Input
                id={inputId}
                className={cn(
                    "w-full",
                    error && "border-destructive focus-visible:ring-destructive",
                    className
                )}
                {...props}
            />
            {description && (
                <p className="text-sm text-muted-foreground">{description}</p>
            )}
            {error && (
                <p className="text-sm font-medium text-destructive">{error}</p>
            )}
        </div>
    );
}; 