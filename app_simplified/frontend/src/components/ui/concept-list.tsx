import React, { useState } from 'react';
import { cn } from '../../lib/utils';
import { Button } from './button';
import { Input } from './input';
import { X } from 'lucide-react';

interface ConceptListProps {
    concepts: string[];
    onChange: (concepts: string[]) => void;
    disabled?: boolean;
    onConceptClick?: (concept: string) => void;
    selectedConcept?: string | null;
    availableConcepts?: string[];
}

export const ConceptList: React.FC<ConceptListProps> = ({
    concepts,
    onChange,
    disabled,
    onConceptClick,
    selectedConcept,
    availableConcepts
}) => {
    const [newConcept, setNewConcept] = useState('');

    const handleAddConcept = () => {
        if (newConcept.trim() && !concepts.includes(newConcept.trim())) {
            if (availableConcepts && !availableConcepts.includes(newConcept.trim())) {
                return;
            }
            onChange([...concepts, newConcept.trim()]);
            setNewConcept('');
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleAddConcept();
        }
    };

    const handleRemoveConcept = (conceptToRemove: string) => {
        onChange(concepts.filter(concept => concept !== conceptToRemove));
    };

    return (
        <div className="space-y-4">
            <div className="flex space-x-2">
                <Input
                    value={newConcept}
                    onChange={(e) => setNewConcept(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Add a concept"
                    disabled={disabled}
                    className="flex-1"
                />
                <Button
                    onClick={handleAddConcept}
                    disabled={disabled || !newConcept.trim()}
                    variant="outline"
                >
                    Add
                </Button>
            </div>

            {concepts.length > 0 ? (
                <div className="space-y-2">
                    {concepts.map((concept) => (
                        <div
                            key={concept}
                            className={cn(
                                "flex items-center justify-between p-2 rounded-md border",
                                selectedConcept === concept && "bg-primary/10 border-primary",
                                "hover:bg-accent"
                            )}
                        >
                            <span
                                className="flex-1 cursor-pointer"
                                onClick={() => onConceptClick?.(concept)}
                            >
                                {concept}
                            </span>
                            {!disabled && (
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => handleRemoveConcept(concept)}
                                    className="h-8 w-8"
                                >
                                    <X className="h-4 w-4" />
                                </Button>
                            )}
                        </div>
                    ))}
                </div>
            ) : (
                <p className="text-sm text-muted-foreground text-center py-2">
                    No concepts added yet
                </p>
            )}
        </div>
    );
}; 