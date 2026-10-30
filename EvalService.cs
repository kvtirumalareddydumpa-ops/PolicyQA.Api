public record EvalCase(string Question, bool Answerable, int? ExpectedPage, string[]? ExpectedContains);

public record EvalCaseResult(
    string Question, bool RetrievalHit, bool AnswerCorrect, bool RefusalCorrect,
    double TopScore, bool SkippedLlm, string Answer);

/// <summary>
/// Runs a labeled evaluation set through the real RagService and scores three layers separately:
/// retrieval hit rate (search), answer correctness (generation), refusal accuracy (guardrail/threshold).
/// "Contains" checks are simple and cheap but brittle ("fifteen" vs "15"); at scale, add LLM-as-a-judge.
/// </summary>
public class EvalService(RagService rag, IConfiguration config)
{
    public async Task<object> RunAsync(IReadOnlyList<EvalCase> cases, CancellationToken ct = default)
    {
        var results = new List<EvalCaseResult>();
        int totalIn = 0, totalOut = 0;
        decimal totalCost = 0;
        int delayMs = config.GetValue("Eval:DelayBetweenCallsMs", 1000);

        foreach (var c in cases)
        {
            var r = await rag.AskAsync(c.Question, ct);
            bool refused = r.Answer.Contains("couldn't find", StringComparison.OrdinalIgnoreCase);

            // Retrieval: was the expected page among the sources?
            bool retrievalHit = !c.Answerable || c.ExpectedPage is null
                || r.Sources.Any(s => s.PageNumber == c.ExpectedPage);

            // Generation: does the answer contain every expected fact?
            bool answerCorrect = !c.Answerable || (!refused
                && (c.ExpectedContains ?? Array.Empty<string>())
                    .All(e => r.Answer.Contains(e, StringComparison.OrdinalIgnoreCase)));

            // Guardrail: refuse only when the question is unanswerable
            bool refusalCorrect = refused == !c.Answerable;

            results.Add(new EvalCaseResult(c.Question, retrievalHit, answerCorrect, refusalCorrect,
                r.Sources.FirstOrDefault()?.Score ?? 0, r.SkippedLlm, r.Answer));

            totalIn += r.InputTokens;
            totalOut += r.OutputTokens;
            totalCost += r.EstimatedCostUsd;

            if (delayMs > 0)
                await Task.Delay(delayMs, ct); // stay under free-tier rate limits
        }

        var answerable = results.Where((_, i) => cases[i].Answerable).ToList();

        static double Pct(int n, int d) => d == 0 ? 0 : Math.Round(100.0 * n / d, 1);

        return new
        {
            summary = new
            {
                totalCases = cases.Count,
                minScoreUsed = config.GetValue("Rag:MinScore", 0.0),
                retrievalHitRatePct = Pct(answerable.Count(r => r.RetrievalHit), answerable.Count),
                answerCorrectnessPct = Pct(answerable.Count(r => r.AnswerCorrect), answerable.Count),
                refusalAccuracyPct = Pct(results.Count(r => r.RefusalCorrect), results.Count),
                skippedLlmCalls = results.Count(r => r.SkippedLlm),
                totalInputTokens = totalIn,
                totalOutputTokens = totalOut,
                estimatedCostUsd = Math.Round(totalCost, 6)
            },
            cases = results
        };
    }
}
