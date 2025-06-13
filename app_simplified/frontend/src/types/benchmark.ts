export interface BenchmarkMetadata {
  id?: number;
  domain: string;
  data?: Record<string, any>;
  table_names: Record<string, string | null>;
  configuration: Record<string, any>;
  database_config: Record<string, any>;
  time_stamp: string;
  created_at?: string; // ISO date string
}

export interface BenchmarkMetadataResponse {
  [tableName: string]: BenchmarkMetadata[];
} 