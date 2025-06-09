import React, { useState } from 'react';
import { Button } from '../ui/button';
import { X } from 'lucide-react';

interface KeywordPair {
    original: string;
    replacement: string;
}

interface KeywordReplacementPairProps {
    pairs: KeywordPair[];
    onPairsChange: (pairs: KeywordPair[]) => void;
    stemConcept?: string;
    branchConcept?: string;
}

const KeywordReplacementPair: React.FC<KeywordReplacementPairProps> = ({ 
    pairs = [], 
    onPairsChange,
    stemConcept,
    branchConcept 
}) => {
    const [newOriginal, setNewOriginal] = useState('');
    const [newReplacement, setNewReplacement] = useState('');

    const handleAddPair = () => {
        if (newOriginal && newReplacement) {
            onPairsChange([...pairs, { original: newOriginal, replacement: newReplacement }]);
            setNewOriginal('');
            setNewReplacement('');
        }
    };

    const handleRemovePair = (index: number) => {
        onPairsChange(pairs.filter((_, i) => i !== index));
    };

    return (
        <div className="space-y-4">
            {stemConcept && branchConcept && (
                <div className="text-sm text-gray-500 mb-2">
                    Configuring keyword replacements from <span className="font-medium">{stemConcept}</span> to <span className="font-medium">{branchConcept}</span>
                </div>
            )}
            <div className="flex gap-4 items-end">
                <div className="flex-1 space-y-2">
                    <label className="text-sm font-medium">Original Keyword</label>
                    <input
                        type="text"
                        value={newOriginal}
                        onChange={(e) => setNewOriginal(e.target.value)}
                        placeholder="Enter original keyword"
                        className="w-full p-2 border rounded-md"
                    />
                </div>

                <div className="flex-1 space-y-2">
                    <label className="text-sm font-medium">Replacement Keyword</label>
                    <input
                        type="text"
                        value={newReplacement}
                        onChange={(e) => setNewReplacement(e.target.value)}
                        placeholder="Enter replacement keyword"
                        className="w-full p-2 border rounded-md"
                    />
                </div>

                <Button
                    onClick={handleAddPair}
                    disabled={!newOriginal || !newReplacement}
                    className="mb-0.5"
                >
                    Add Pair
                </Button>
            </div>

            <div className="space-y-2">
                {pairs.map((pair, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 rounded-md">
                        <span className="flex-1">
                            <span className="font-medium">{pair.original}</span>
                            <span className="mx-2">→</span>
                            <span>{pair.replacement}</span>
                        </span>
                        <Button
                            onClick={() => handleRemovePair(index)}
                            variant="ghost"
                            size="sm"
                        >
                            <X className="h-4 w-4" />
                        </Button>
                    </div>
                ))}
                {pairs.length === 0 && (
                    <div className="text-sm text-gray-500 italic text-center py-4">
                        No keyword replacements added yet
                    </div>
                )}
            </div>
        </div>
    );
};

export { KeywordReplacementPair }; 