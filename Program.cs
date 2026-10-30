using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Gemini HTTP clients. The API key comes from User Secrets locally (Key Vault in production), never from Git.
Action<HttpClient> configureGemini = client =>
{
    client.BaseAddress = new Uri(builder.Configuration["Gemini:BaseUrl"]!);
    client.DefaultRequestHeaders.Add("x-goog-api-key", builder.Configuration["Gemini:ApiKey"]);
    client.Timeout = TimeSpan.FromSeconds(builder.Configuration.GetValue("Gemini:TimeoutSeconds", 30));
};

// Typed HttpClients via IHttpClientFactory (avoids socket exhaustion)
builder.Services.AddHttpClient<ILlmClient, GeminiClient>(configureGemini);
builder.Services.AddHttpClient<IEmbeddingClient, GeminiEmbeddingClient>(configureGemini);

// Singletons hold shared state; scoped services live for one request
builder.Services.AddSingleton<IVectorStore, InMemoryVectorStore>();
builder.Services.AddSingleton<UsageTracker>();
builder.Services.AddScoped<IngestionService>();
builder.Services.AddScoped<RagService>();
builder.Services.AddScoped<EvalService>();

builder.Services.AddProblemDetails();

var app = builder.Build();

app.UseExceptionHandler(); // unexpected errors → clean ProblemDetails, no stack traces to users

// Maps AI-provider failures to clean HTTP errors (502 / 504) and logs the details
async Task<IResult> CallProvider(Func<Task<IResult>> action, CancellationToken ct)
{
    try
    {
        return await action();
    }
    catch (HttpRequestException ex)
    {
        app.Logger.LogError(ex, "AI provider returned an error");
        return Results.Problem("The AI provider returned an error. Please try again.",
            statusCode: StatusCodes.Status502BadGateway);
    }
    catch (TaskCanceledException ex) when (!ct.IsCancellationRequested)
    {
        app.Logger.LogWarning(ex, "AI provider timed out");
        return Results.Problem("The AI provider timed out. Please try again.",
            statusCode: StatusCodes.Status504GatewayTimeout);
    }
}

app.MapGet("/", () => Results.Ok(new
{
    name = "PolicyQA API",
    endpoints = new[] { "POST /api/documents", "POST /api/ask", "GET /api/usage", "POST /api/eval" }
}));

// Ingestion: upload a PDF → chunk → embed → store
app.MapPost("/api/documents", async (IFormFile file, IngestionService ingestion, CancellationToken ct) =>
{
    if (file.Length == 0 || !file.FileName.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
        return Results.BadRequest("Please upload a non-empty PDF.");

    return await CallProvider(async () =>
    {
        try
        {
            await using var stream = file.OpenReadStream();
            return Results.Ok(await ingestion.IngestPdfAsync(stream, file.FileName, ct));
        }
        catch (IngestionException ex)
        {
            return Results.BadRequest(ex.Message);
        }
    }, ct);
}).DisableAntiforgery();

// Question answering: retrieve → threshold → grounded prompt → answer with citations
app.MapPost("/api/ask", async (ChatRequest request, RagService rag, IVectorStore store, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(request.Question))
        return Results.BadRequest("Question is required.");
    if (request.Question.Length > 1000)
        return Results.BadRequest("Question is too long (max 1000 characters).");
    if (store.Count == 0)
        return Results.BadRequest("Upload a document first.");

    return await CallProvider(async () => Results.Ok(await rag.AskAsync(request.Question, ct)), ct);
});

// Running totals: tokens, estimated cost, skipped LLM calls
app.MapGet("/api/usage", (UsageTracker usage) => Results.Ok(usage.Snapshot()));

// Labeled evaluation: runs evalset.json through the same RagService users hit
app.MapPost("/api/eval", async (EvalService evals, IVectorStore store, IWebHostEnvironment env, CancellationToken ct) =>
{
    if (store.Count == 0)
        return Results.BadRequest("Upload a document first.");

    var path = Path.Combine(env.ContentRootPath, "evalset.json");
    if (!File.Exists(path))
        return Results.BadRequest("evalset.json not found in the project folder.");

    var json = await File.ReadAllTextAsync(path, ct);
    var cases = JsonSerializer.Deserialize<List<EvalCase>>(json, new JsonSerializerOptions(JsonSerializerDefaults.Web));
    if (cases is null || cases.Count == 0)
        return Results.BadRequest("evalset.json has no cases.");

    return await CallProvider(async () => Results.Ok(await evals.RunAsync(cases, ct)), ct);
});

app.Run();

record ChatRequest(string Question);
