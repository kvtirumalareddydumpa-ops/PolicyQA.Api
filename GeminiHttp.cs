using System.Net;
using System.Text.Json;

/// <summary>
/// Shared HTTP helper for all Gemini calls.
/// Retries transient errors (429, 503, 504) with exponential backoff: 1s, then 2s.
/// </summary>
public static class GeminiHttp
{
    private static readonly HttpStatusCode[] TransientStatusCodes =
    [
        HttpStatusCode.TooManyRequests,     // 429: rate limited
        HttpStatusCode.ServiceUnavailable,  // 503: provider overloaded
        HttpStatusCode.GatewayTimeout       // 504: upstream timeout
    ];

    public static async Task<JsonDocument> PostJsonAsync(
        HttpClient http, string path, object body, CancellationToken ct, int maxRetries = 2)
    {
        for (int attempt = 0; ; attempt++)
        {
            using var response = await http.PostAsJsonAsync(path, body, ct);
            var json = await response.Content.ReadAsStringAsync(ct);

            if (response.IsSuccessStatusCode)
                return JsonDocument.Parse(json);

            if (attempt < maxRetries && TransientStatusCodes.Contains(response.StatusCode))
            {
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), ct);
                continue;
            }

            // Full provider message goes to the logs; the API returns a generic error to users
            throw new HttpRequestException(
                $"Gemini error {(int)response.StatusCode}: {json}", null, response.StatusCode);
        }
    }
}
