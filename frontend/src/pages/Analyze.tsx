import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Upload, 
  Settings, 
  Sparkles, 
  HelpCircle, 
  Play, 
  FileText,
  AlertTriangle,
  Clock,
  Car,
  UserCheck,
  CheckCircle2,
  Trash2
} from 'lucide-react'
import { analyzeImage, AnalysisResult } from '../api'

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  
  // Options
  const [enhanceContrast, setEnhanceContrast] = useState(false)
  const [checkStopLine, setCheckStopLine] = useState(false)
  const [detectParking, setDetectParking] = useState(true)
  const [flowDirection, setFlowDirection] = useState('none')
  const [cameraId, setCameraId] = useState('CAM_DOWNTOWN_04')

  // Results
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selected = acceptedFiles[0]
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResult(null)
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    maxFiles: 1,
    disabled: loading
  })

  const handleClear = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyzeImage(file, {
        enhance_contrast: enhanceContrast,
        check_stop_line: checkStopLine,
        detect_parking: detectParking,
        flow_direction: flowDirection,
        camera_id: cameraId
      })
      setResult(data)
    } catch (err: any) {
      console.error(err)
      setError(
        err.response?.data?.detail || 
        'An error occurred during analysis. Please ensure the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  const formatViolationType = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Evidence Analysis Engine</h1>
          <p className="page-subtitle">Process traffic imagery to extract detections and violations</p>
        </div>
      </div>

      <div className="grid-3" style={{ gridTemplateColumns: '280px 1fr 1fr', alignItems: 'stretch' }}>
        
        {/* Left Column: Settings / Options */}
        <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: 12 }}>
            <Settings size={18} className="v-stopline" /> Configuration
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Camera Source</label>
            <select 
              className="input" 
              value={cameraId} 
              onChange={e => setCameraId(e.target.value)}
              disabled={loading}
            >
              <option value="CAM_DOWNTOWN_04">CAM_DOWNTOWN_04 (Main St)</option>
              <option value="CAM_HIGHWAY_12">CAM_HIGHWAY_12 (Expressway)</option>
              <option value="CAM_INTERSECT_09">CAM_INTERSECT_09 (Broadway)</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Pipeline Settings</label>
            
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: '0.85rem' }}>
              <input 
                type="checkbox" 
                checked={enhanceContrast} 
                onChange={e => setEnhanceContrast(e.target.checked)}
                disabled={loading}
                style={{ width: 16, height: 16, accentColor: 'var(--amber)' }}
              />
              <div>
                <div>Enhance Contrast (CLAHE)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Improves detection under shadows/glare</div>
              </div>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: '0.85rem' }}>
              <input 
                type="checkbox" 
                checked={checkStopLine} 
                onChange={e => setCheckStopLine(e.target.checked)}
                disabled={loading}
                style={{ width: 16, height: 16, accentColor: 'var(--amber)' }}
              />
              <div>
                <div>Verify Stop-line Crossing</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Detect encroachment over crossing boundaries</div>
              </div>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: '0.85rem' }}>
              <input 
                type="checkbox" 
                checked={detectParking} 
                onChange={e => setDetectParking(e.target.checked)}
                disabled={loading}
                style={{ width: 16, height: 16, accentColor: 'var(--amber)' }}
              />
              <div>
                <div>Detect Illegal Parking</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Flags vehicles at road edges with no driver nearby</div>
              </div>
            </label>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
              <label style={{ fontSize: '0.85rem' }}>Traffic Flow Direction</label>
              <select 
                value={flowDirection} 
                onChange={(e) => setFlowDirection(e.target.value)}
                disabled={loading}
                className="input"
                style={{ fontSize: '0.85rem', padding: '6px 12px' }}
              >
                <option value="none">Wrong-Side: Off</option>
                <option value="right">Expected: Keep Right</option>
                <option value="left">Expected: Keep Left</option>
              </select>
            </div>
          </div>

          <div style={{ marginTop: 'auto', paddingTop: 16, borderTop: '1px solid var(--border)' }}>
            {preview && (
              <div style={{ display: 'flex', gap: 10 }}>
                <button 
                  className="btn btn-secondary btn-sm" 
                  onClick={handleClear}
                  disabled={loading}
                  style={{ flex: 1 }}
                >
                  <Trash2 size={14} /> Clear
                </button>
                <button 
                  className="btn btn-primary btn-sm" 
                  onClick={handleAnalyze}
                  disabled={loading}
                  style={{ flex: 2 }}
                >
                  {loading ? (
                    <>
                      <div className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5, marginRight: 4 }} /> Analyzing...
                    </>
                  ) : (
                    <>
                      <Play size={14} style={{ marginRight: 4 }} /> Process
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Center / Right Columns: Image preview & results */}
        <div style={{ gridColumn: 'span 2', display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* File Selector */}
          {!preview && (
            <div {...getRootProps()} className={`drop-zone ${isDragActive ? 'drag-over' : ''}`} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <input {...getInputProps()} />
              <div className="drop-zone-icon">📸</div>
              <div className="drop-zone-title">Drag & drop traffic image here</div>
              <div className="drop-zone-sub">Supports PNG, JPG, JPEG up to 10MB</div>
            </div>
          )}

          {/* Loader bar */}
          {loading && (
            <div className="card glass" style={{ textAlign: 'center', padding: '40px 20px' }}>
              <div className="spinner" style={{ marginBottom: 16 }} />
              <h3>Running Computer Vision Models</h3>
              <p style={{ marginTop: 8, fontSize: '0.85rem' }}>Detecting vehicles, helmets, seatbelts, and license plates...</p>
              <div className="loading-bar" style={{ marginTop: 24, maxWidth: 300, marginLeft: 'auto', marginRight: 'auto' }}>
                <div className="loading-bar-inner" />
              </div>
            </div>
          )}

          {/* Error display */}
          {error && (
            <div className="alert alert-error" style={{ margin: '10px 0' }}>
              <AlertTriangle size={18} />
              <div>
                <strong>Analysis Failed</strong>
                <p style={{ fontSize: '0.8rem', marginTop: 4, color: 'inherit' }}>{error}</p>
              </div>
            </div>
          )}

          {/* Image Comparison */}
          {preview && !loading && (
            <div className="image-grid">
              <div className="image-box">
                <div className="image-box-header">Original Image</div>
                <img src={preview} alt="Original traffic capture" />
              </div>

              <div className="image-box" style={{ borderColor: result ? 'var(--amber-dark)' : 'var(--border)' }}>
                <div className="image-box-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Annotated Evidence</span>
                  {result && <span className="v-helmet" style={{ fontWeight: 800 }}>AI Detections Active</span>}
                </div>
                {result?.annotated_image_url ? (
                  <img src={result.annotated_image_url} alt="AI Detections" />
                ) : (
                  <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#05080f', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Awaiting pipeline execution...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Results Summary Cards */}
          {result && !loading && (
            <motion.div 
              className="grid-2" 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {/* Detections metadata */}
              <div className="card glass">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, fontSize: '1rem' }}>
                  <Car size={18} className="v-wrongside" /> Vehicle Context
                </h3>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Classified Type</span>
                    <strong style={{ color: 'var(--text-primary)', textTransform: 'capitalize' }}>{result.vehicle_type}</strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Inference Speed</span>
                    <strong style={{ color: 'var(--green-light)', fontFamily: 'var(--font-mono)' }}>{result.inference_time_ms.toFixed(1)} ms</strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Person Count</span>
                    <strong style={{ color: 'var(--text-primary)' }}>{result.person_count ?? 1}</strong>
                  </div>

                  {result.license_plate && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Extracted OCR License Plate</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <div className="plate-display">{result.license_plate}</div>
                        <div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>OCR Confidence</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            {result.plate_confidence ? `${Math.round(result.plate_confidence * 100)}%` : 'Rule-based'}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Violations List */}
              <div className="card glass">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, fontSize: '1rem' }}>
                  <AlertTriangle size={18} className="v-helmet" /> Violations Found ({result.violation_count})
                </h3>

                {result.violation_count === 0 ? (
                  <div className="empty-state" style={{ padding: '20px 0' }}>
                    <CheckCircle2 size={32} className="v-green" style={{ marginBottom: 8 }} />
                    <p style={{ fontSize: '0.85rem' }}>No traffic violation detected in this image.</p>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {result.violations.map((violation, idx) => (
                      <div 
                        key={idx} 
                        style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'space-between', 
                          padding: '10px 12px', 
                          background: 'rgba(255,255,255,0.03)', 
                          border: '1px solid var(--border)', 
                          borderRadius: 'var(--radius-sm)' 
                        }}
                      >
                        <div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {formatViolationType(violation.violation_type)}
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
                            {violation.description}
                          </div>
                        </div>
                        <span className={`badge ${
                          violation.severity === 'critical' ? 'badge-red' : 
                          violation.severity === 'high' ? 'badge-red' : 'badge-amber'
                        }`}>
                          {violation.severity}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          )}

        </div>
      </div>
    </div>
  )
}
