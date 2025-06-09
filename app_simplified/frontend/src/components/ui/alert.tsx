import React from 'react';

interface AlertProps {
    children: React.ReactNode;
    variant?: 'default' | 'destructive';
    className?: string;
}

export const Alert: React.FC<AlertProps> = ({ 
    children, 
    variant = 'default',
    className = ''
}) => {
    const baseStyles = "p-4 rounded-md text-sm";
    const variantStyles = {
        default: "bg-blue-50 text-blue-700 border border-blue-200",
        destructive: "bg-red-50 text-red-700 border border-red-200"
    };

    return (
        <div className={`${baseStyles} ${variantStyles[variant]} ${className}`}>
            {children}
        </div>
    );
}; 