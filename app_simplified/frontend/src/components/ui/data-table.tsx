import * as React from "react"
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Box,
} from '@mui/material';

interface DataFrameFormat {
  columns: string[];
  data: Record<string, any>[];
  index?: (string | number)[];
}

interface DataTableProps {
  data: Record<string, any> | DataFrameFormat;
  className?: string;
}

const formatValue = (value: any): React.ReactNode => {
  if (value === null || value === undefined) {
    return '-';
  }
  
  if (typeof value === 'object') {
    return (
      <Box sx={{ maxWidth: 300, overflow: 'auto' }}>
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: '0.875rem' }}>
          {JSON.stringify(value, null, 2)}
        </pre>
      </Box>
    );
  }
  
  return String(value);
};

const isDataFrameFormat = (data: any): data is DataFrameFormat => {
  return data && 
         Array.isArray(data.columns) && 
         Array.isArray(data.data) &&
         data.columns.length > 0;
};

export const DataTable = React.forwardRef<HTMLDivElement, DataTableProps>(
  ({ data, className, ...props }, ref) => {
    if (isDataFrameFormat(data)) {
      const { columns, data: rows, index } = data;
      
      return (
        <TableContainer component={Paper} sx={{ mb: 2 }} ref={ref} {...props}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Index</TableCell>
                {columns.map((column) => (
                  <TableCell key={column}>{column}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row, i) => (
                <TableRow key={i}>
                  <TableCell>{index?.[i] ?? i}</TableCell>
                  {columns.map((column) => (
                    <TableCell key={`${i}-${column}`}>
                      {formatValue(row[column])}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      );
    }

    // Original format handling
    const parsedData: Record<string, any> = {};
    Object.entries(data).forEach(([key, value]) => {
      if (typeof value === 'string') {
        try {
          parsedData[key] = JSON.parse(value);
        } catch (e) {
          parsedData[key] = value;
        }
      } else {
        parsedData[key] = value;
      }
    });

    const allIndices = new Set<number>();
    Object.values(parsedData).forEach(value => {
      if (typeof value === 'object' && value !== null) {
        Object.keys(value).forEach(key => {
          if (!isNaN(Number(key))) {
            allIndices.add(Number(key));
          }
        });
      }
    });

    const sortedIndices = Array.from(allIndices).sort((a, b) => a - b);

    return (
      <TableContainer component={Paper} sx={{ mb: 2 }} ref={ref} {...props}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Index</TableCell>
              {Object.keys(parsedData).map((columnName) => (
                <TableCell key={columnName}>{columnName}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedIndices.map((index) => (
              <TableRow key={index}>
                <TableCell>{index}</TableCell>
                {Object.keys(parsedData).map((columnName) => {
                  const columnData = parsedData[columnName];
                  const value = typeof columnData === 'object' && columnData !== null
                    ? columnData[index]
                    : columnData;
                  return (
                    <TableCell key={`${columnName}-${index}`}>
                      {formatValue(value)}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  }
);

DataTable.displayName = "DataTable"; 