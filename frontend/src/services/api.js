import axios from 'axios';

const API_BASE = '/api';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const api = {
  // System Health
  getHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Inspect Single Image
  inspectImage: async (file, productType = 'metal_sheet', confThreshold = 0.25, alpha = 0.45) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/inspect/image', formData, {
      params: {
        product_type: productType,
        conf_threshold: confThreshold,
        alpha: alpha,
      },
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  // Stream Frame / Live Camera
  inspectStreamFrame: async (imageBase64, confThreshold = 0.25, enableAnomaly = true, renderMode = 'annotated') => {
    const res = await apiClient.post('/inspect/stream-frame', {
      image_base64: imageBase64,
      conf_threshold: confThreshold,
      enable_anomaly: enableAnomaly,
      render_mode: renderMode,
    });
    return res.data;
  },

  // Sample Gallery
  getSamples: async () => {
    const res = await apiClient.get('/samples');
    return res.data.samples || [];
  },

  // Executive Dashboard Analytics
  getAnalytics: async () => {
    const res = await apiClient.get('/analytics');
    return res.data;
  },

  // History Log
  getHistory: async (limit = 50, offset = 0, verdict = null) => {
    const res = await apiClient.get('/history', {
      params: { limit, offset, verdict },
    });
    return res.data;
  },

  // Delete Record
  deleteRecord: async (recordId) => {
    const res = await apiClient.delete(`/history/${recordId}`);
    return res.data;
  },

  // Clear History
  clearHistory: async () => {
    const res = await apiClient.post('/history/clear');
    return res.data;
  },

  // Export CSV URL
  getExportCsvUrl: () => `${API_BASE}/export/csv`,

  // Model Metrics
  getModelMetrics: async () => {
    const res = await apiClient.get('/model/metrics');
    return res.data;
  },
};
