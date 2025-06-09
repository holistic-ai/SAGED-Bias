// API configuration
export const API_HOST = 'http://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
    BENCHMARK: {
        BUILD: `${API_HOST}/benchmark/build`,
    },
    FILES: {
        UPLOAD: (domain: string) => `${API_HOST}/files/upload/${domain}`,
        LIST: (domain: string) => `${API_HOST}/files/${domain}`,
        CONTENT: (domain: string, filename: string) => `${API_HOST}/files/content/${domain}/${filename}`,
        CLEANUP: (domain: string) => `${API_HOST}/files/${domain}`,
    },
} as const; 