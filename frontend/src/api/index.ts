import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 300_000, // 5 min for large batch uploads or slow CPU inference
})

export interface AnalysisResult {
  violations: DetectedViolation[]
  vehicle_type: string
  license_plate?: string
  plate_confidence?: number
  person_count?: number
  all_detections: BoundingBox[]
  annotated_image_url?: string
  inference_time_ms: number
  violation_count: number
}

export interface DetectedViolation {
  violation_type: string
  confidence: number
  severity: string
  description: string
  bbox?: BoundingBox
}

export interface BoundingBox {
  x1: number; y1: number; x2: number; y2: number
  confidence: number; class_name: string; class_id: number
}

export interface ViolationRecord {
  id: number
  timestamp: string
  violation_type: string
  violation_types?: string[]
  confidence: number
  severity: string
  vehicle_type: string
  license_plate?: string
  plate_confidence?: number
  original_image_path?: string
  annotated_image_path?: string
  camera_id?: string
  location?: string
  status: string
  inference_time_ms?: number
  person_count?: number
}

export interface PaginatedViolations {
  items: ViolationRecord[]
  total: number
  page: number
  page_size: number
  pages: number
}

export interface AnalyticsSummary {
  total_violations: number
  today_violations: number
  pending_review: number
  resolution_rate: number
  top_violations: { violation_type: string; count: number; percentage: number }[]
  vehicle_distribution: Record<string, number>
  avg_confidence: number
}

// ── Analysis ─────────────────────────────────────────────────────────────────

export const analyzeImage = async (
  file: File,
  options?: { enhance_contrast?: boolean; check_stop_line?: boolean; detect_parking?: boolean; flow_direction?: string; camera_id?: string }
): Promise<AnalysisResult> => {
  const form = new FormData()
  form.append('file', file)
  if (options?.enhance_contrast !== undefined) form.append('enhance_contrast', String(options.enhance_contrast))
  if (options?.check_stop_line !== undefined) form.append('check_stop_line', String(options.check_stop_line))
  if (options?.detect_parking !== undefined) form.append('detect_parking', String(options.detect_parking))
  if (options?.flow_direction !== undefined) form.append('flow_direction', options.flow_direction)
  if (options?.camera_id) form.append('camera_id', options.camera_id)
  const { data } = await api.post<AnalysisResult>('/analyze/image', form)
  return data
}

export const analyzeBatch = async (files: File[]): Promise<any> => {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  const { data } = await api.post('/analyze/batch', form)
  return data
}

// ── Violations ────────────────────────────────────────────────────────────────

export const fetchViolations = async (params?: {
  page?: number; page_size?: number; violation_type?: string
  status?: string; license_plate?: string
}): Promise<PaginatedViolations> => {
  const { data } = await api.get('/violations', { params })
  return data
}

export const fetchViolation = async (id: number): Promise<ViolationRecord> => {
  const { data } = await api.get(`/violations/${id}`)
  return data
}

export const updateViolation = async (id: number, update: { status?: string; notes?: string }) => {
  const { data } = await api.patch(`/violations/${id}`, update)
  return data
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export const fetchSummary = async (): Promise<AnalyticsSummary> => {
  const { data } = await api.get('/analytics/summary')
  return data
}

export const fetchTrends = async (days = 7) => {
  const { data } = await api.get('/analytics/trends', { params: { days } })
  return data
}

export const fetchByType = async () => {
  const { data } = await api.get('/analytics/by-type')
  return data
}

export const fetchMetrics = async () => {
  const { data } = await api.get('/analytics/metrics')
  return data
}

export const fetchHealth = async () => {
  const { data } = await axios.get('/health')
  return data
}

export const downloadCSV = () => {
  window.open('/api/v1/analytics/export/csv', '_blank')
}

// ── Vehicles ──────────────────────────────────────────────────────────────────

export const searchVehicles = async (plate: string) => {
  const { data } = await api.get('/vehicles/search', { params: { plate } })
  return data
}

export const fetchVehicle = async (id: number) => {
  const { data } = await api.get(`/vehicles/${id}`)
  return data
}

export const fetchVehicleViolations = async (id: number) => {
  const { data } = await api.get(`/vehicles/${id}/violations`)
  return data
}

// ── Challans ──────────────────────────────────────────────────────────────────

export const fetchChallans = async (status?: string) => {
  const { data } = await api.get('/challans', { params: { status } })
  return data
}

export const generateChallan = async (violationId: number) => {
  const { data } = await api.post('/challans/generate', { violation_id: violationId })
  return data
}

export const payChallan = async (challanId: number) => {
  const { data } = await api.put(`/challans/${challanId}/pay`)
  return data
}

// ── Cameras ───────────────────────────────────────────────────────────────────

export const fetchCameras = async () => {
  const { data } = await api.get('/cameras')
  return data
}

export const createCamera = async (payload: any) => {
  const { data } = await api.post('/cameras', payload)
  return data
}

export const updateCamera = async (id: number, payload: any) => {
  const { data } = await api.put(`/cameras/${id}`, payload)
  return data
}

export const deleteCamera = async (id: number) => {
  const { data } = await api.delete(`/cameras/${id}`)
  return data
}

// ── Settings & Reports ────────────────────────────────────────────────────────

export const fetchSettings = async () => {
  const { data } = await api.get('/settings')
  return data
}

export const updateSettings = async (payload: any) => {
  const { data } = await api.put('/settings', payload)
  return data
}

export const generateReport = (type: string) => {
  window.open(`http://127.0.0.1:8000/api/v1/reports/generate?type=${type}`, '_blank')
}

export default api
