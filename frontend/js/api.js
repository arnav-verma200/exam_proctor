const API_BASE = window.location.origin;

export function getApiBase() {
  return API_BASE;
}

export async function apiRequest(path, options = {}) {
  const url = API_BASE + path;
  const config = {
    credentials: 'include',
    headers: {},
    ...options,
  };

  const isFormData = config.body instanceof FormData;
  const isBlob = config.body instanceof Blob;

  if (!isFormData && !isBlob && config.body && typeof config.body === 'object') {
    config.headers['Content-Type'] = 'application/json';
    config.body = JSON.stringify(config.body);
  }

  const res = await fetch(url, config);
  const contentType = res.headers.get('content-type') || '';

  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      if (contentType.includes('application/json')) {
        const data = await res.json();
        msg = data.message || msg;
      }
    } catch {}
    throw new Error(msg);
  }

  if (contentType.includes('application/json')) {
    return res.json();
  }

  return res;
}

export async function checkApiHealth() {
  try {
    await apiRequest('/api/exam/info');
    return true;
  } catch {
    return false;
  }
}
