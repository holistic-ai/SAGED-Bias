import React from 'react';
import { cn } from '../../lib/utils';

interface AlertProps {
    children: React.ReactNode;
    variant?: 'default' | 'destructive' | 'success' | 'warning' | 'info';
    className?: string;
}

export const Alert: React.FC<AlertProps> = ({ 
    children, 
    variant = 'default',
    className = ''
}) => {
    const baseStyles = "p-4 rounded-lg text-sm shadow-sm backdrop-blur-sm transition-all duration-200";
    const variantStyles = {
        default: "bg-blue-50/90 text-blue-700 border border-blue-200 hover:bg-blue-50",
        destructive: "bg-red-50/90 text-red-700 border border-red-200 hover:bg-red-50",
        success: "bg-green-50/90 text-green-700 border border-green-200 hover:bg-green-50",
        warning: "bg-yellow-50/90 text-yellow-700 border border-yellow-200 hover:bg-yellow-50",
        info: "bg-slate-50/90 text-slate-700 border border-slate-200 hover:bg-slate-50"
    };

    return (
        <div className={cn(baseStyles, variantStyles[variant], className)}>
            {children}
        </div>
    );
}; 