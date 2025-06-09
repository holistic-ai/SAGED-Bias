import React, { useRef } from 'react';
import { Button } from './button';

interface FileUploadListProps {
    files: File[];
    onFilesChange: (files: File[]) => void;
    disabled?: boolean;
    accept?: string;
    label?: string;
    emptyMessage?: string;
}

const FileUploadList: React.FC<FileUploadListProps> = ({
    files,
    onFilesChange,
    disabled = false,
    accept = ".txt,.csv,.json",
    label = "Upload Files",
    emptyMessage = "No files uploaded. Click 'Upload Files' to add files."
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            const newFiles = Array.from(e.target.files);
            onFilesChange([...files, ...newFiles]);
        }
    };

    const handleRemoveFile = (index: number) => {
        onFilesChange(files.filter((_, i) => i !== index));
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <Button
                    onClick={handleUploadClick}
                    variant="outline"
                    className="text-sm"
                    disabled={disabled}
                >
                    {label}
                </Button>
            </div>
            
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                multiple
                className="hidden"
                accept={accept}
            />

            <div className="space-y-2 min-h-[100px] border rounded-md p-2">
                {files.length > 0 ? (
                    files.map((file, index) => (
                        <div
                            key={index}
                            className="flex items-center justify-between p-2 bg-gray-50 rounded-md"
                        >
                            <span className="text-sm truncate max-w-[80%]">
                                {file.name}
                            </span>
                            <Button
                                onClick={() => handleRemoveFile(index)}
                                variant="ghost"
                                className="text-red-500 hover:text-red-600"
                            >
                                Remove
                            </Button>
                        </div>
                    ))
                ) : (
                    <div className="text-sm text-gray-500 text-center py-4">
                        {emptyMessage}
                    </div>
                )}
            </div>
        </div>
    );
};

export default FileUploadList; 