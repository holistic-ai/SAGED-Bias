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
                "rounded-lg border border-slate-200 bg-white/95 backdrop-blur-sm text-slate-900 shadow-lg hover:shadow-xl transition-all duration-200",
                className
            )}
            {...props}
        >
            <div className="p-6 space-y-4">
                <div className="space-y-1 border-b border-slate-200 pb-4">
                    <h3 className="text-lg font-semibold leading-none tracking-tight bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                        {title}
                    </h3>
                    {description && (
                        <p className="text-sm text-slate-600 leading-relaxed">
                            {description}
                        </p>
                    )}
                </div>
                <div className="bg-slate-50/50 rounded-lg p-4">
                    {children}
                </div>
            </div>
        </div>
    );
}; 