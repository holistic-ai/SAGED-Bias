import React from 'react';
import { cn } from '../../lib/utils';

interface FormCardProps extends React.HTMLAttributes<HTMLDivElement> {
    title: string;
    description?: string;
    children: React.ReactNode;
}

export const FormCard: React.FC<FormCardProps> = ({
    title,
    description,
    children,
    className,
    ...props
}) => {
    return (
        <div
            className={cn(
                "rounded-lg border bg-card text-card-foreground shadow-sm",
                className
            )}
            {...props}
        >
            <div className="p-6 space-y-4">
                <div className="space-y-1">
                    <h3 className="text-lg font-semibold leading-none tracking-tight">
                        {title}
                    </h3>
                    {description && (
                        <p className="text-sm text-muted-foreground">
                            {description}
                        </p>
                    )}
                </div>
                {children}
            </div>
        </div>
    );
}; 