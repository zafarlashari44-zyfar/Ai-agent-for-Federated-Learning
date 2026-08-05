export async function analyseECG(formData: FormData) {
  const response = await fetch("/api/ecg/analyse", {
    method: "POST",
    body: formData,
    cache: "no-store",
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(
      payload?.error?.message ??
      "ECG analysis failed."
    );
  }

  return payload;
}
