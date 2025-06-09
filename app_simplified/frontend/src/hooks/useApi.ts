import { useState, useCallback } from 'react'
import axios, { AxiosError } from 'axios'

interface UseApiResponse<T> {
  data: T | null
  error: string | null
  loading: boolean
  execute: (url: string, method?: string, body?: any) => Promise<void>
}

export function useApi<T>(): UseApiResponse<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  const execute = useCallback(async (url: string, method = 'GET', body?: any) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await axios({
        method,
        url,
        data: body,
      })
      
      setData(response.data)
    } catch (err) {
      const error = err as AxiosError
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, error, loading, execute }
} 