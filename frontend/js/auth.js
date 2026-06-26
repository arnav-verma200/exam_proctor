import { apiRequest } from './api.js';

export async function candidateLogin(appNo, password) {
  return apiRequest('/api/login', {
    method: 'POST',
    body: { app_number: appNo, password },
  });
}

export async function teacherLogin(username, password) {
  return apiRequest('/api/teacher/login', {
    method: 'POST',
    body: { username, password },
  });
}

export async function teacherLogoutApi() {
  try {
    await apiRequest('/api/teacher/logout', { method: 'POST' });
  } catch {}
}

export async function agreeAndStart() {
  return apiRequest('/api/exam/start', {
    method: 'POST',
    body: { agreed: true },
  });
}

export async function setCandidate(appNo, name) {
  try {
    await fetch(getApiBase() + '/proctor/set_candidate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ app_number: appNo, student_name: name }),
    });
  } catch {}
}

export async function clearCandidate() {
  try {
    await fetch(getApiBase() + '/proctor/clear_candidate', { method: 'POST' });
  } catch {}
}

function getApiBase() {
  return window.location.origin;
}
