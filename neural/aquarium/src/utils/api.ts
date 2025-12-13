export const API_CONFIG = {
  baseURL: process.env.VITE_API_URL || 'http://localhost:8051',
  endpoints: {
    compile: '/api/compile',
    validate: '/api/validate',
    export: '/api/export',
    parse: '/api/parse',
  },
};

export async function compileModel(dslCode: string): Promise<any> {
  const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.compile}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dsl: dslCode }),
  });

  if (!response.ok) {
    throw new Error(`Compilation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function validateDSL(dslCode: string): Promise<any> {
  const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.validate}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dsl: dslCode }),
  });

  if (!response.ok) {
    throw new Error(`Validation failed: ${response.statusText}`);
  }

  return response.json();
}

export async function exportModel(dslCode: string, backend: 'tensorflow' | 'pytorch' | 'onnx'): Promise<any> {
  const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.export}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dsl: dslCode, backend }),
  });

  if (!response.ok) {
    throw new Error(`Export failed: ${response.statusText}`);
  }

  return response.json();
}

export async function parseDSL(dslCode: string): Promise<any> {
  const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.parse}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dsl: dslCode }),
  });

  if (!response.ok) {
    throw new Error(`Parsing failed: ${response.statusText}`);
  }

  return response.json();
}
