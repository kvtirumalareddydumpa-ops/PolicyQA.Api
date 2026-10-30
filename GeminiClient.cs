using System.Text.Json;

public interface ILlmClient
{
    Task<LlmResult> GenerateAsync(string systemInstruction, string input, CancellationToken ct = default);
}

public record LlmResult(string Text, int InputTokens, int OutputTokens);

/// <summary>
/// Calls the Gemini Interactions API. The official docs list no .NET SDK,
/// so this is a typed HttpClient behind the ILlmClient interface.
/// Swapping to Azure OpenAI = a new class implementing ILlmClient + one line in Program.cs.
/// </summary>
public class GeminiClient(HttpClient http, IConfiguration config) : ILlmClient
{
    private readonly string _model = config["Gemini:ChatModel"]!;
    private readonly string _thinkingLevel = config["Gemini:ThinkingLevel"] ?? "low";

    public async Task<LlmResult> GenerateAsync(string systemInstruction, string input, CancellationToken ct = default)
    {
        var body = new
        {
            model = _model,
            system_instruction = systemInstruction,
            input,
            generation_config = new { thinking_level = _thinkingLevel }
        };

        using var doc = await GeminiHttp.PostJsonAsync(http, "interactions", body, ct);
        return ParseResponse(doc.RootElement);
    }

    /// <summary>Public and static so it can be unit tested without calling the API.</summary>
    public static LlmResult ParseResponse(JsonElement root)
    {
        var text = string.Empty;

        // The final answer lives in the "model_output" step(s); "thought" steps are skipped
        if (root.TryGetProperty("steps", out var steps))
        {
            text = string.Concat(
                steps.EnumerateArray()
                    .Where(s => s.TryGetProperty("type", out var type) && type.GetString() == "model_output")
                    .SelectMany(s => s.TryGetProperty("content", out var content)
                        ? content.EnumerateArray()
                        : Enumerable.Empty<JsonElement>())
                    .Where(c => c.TryGetProperty("type", out var type) && type.GetString() == "text")
                    .Select(c => c.GetProperty("text").GetString()));
        }

        int inputTokens = 0, outputTokens = 0;
        if (root.TryGetProperty("usage", out var usage))
        {
            if (usage.TryGetProperty("total_input_tokens", out var i)) inputTokens = i.GetInt32();
            if (usage.TryGetProperty("total_output_tokens", out var o)) outputTokens = o.GetInt32();
        }

        return new LlmResult(text, inputTokens, outputTokens);
    }
}
