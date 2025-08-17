export const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.REACT_APP_API_URL as string) || ;

export async function getStatus(): Promise<{ status: string } | null> {
  if (!API_URL) return null;
  try {
    const res = await fetch(`${API_URL}/api/v1/status`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
