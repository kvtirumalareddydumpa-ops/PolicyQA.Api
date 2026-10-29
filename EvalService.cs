public record EvalCase(string Question, bool Answerable, int? ExpectedPage, string[]? ExpectedContains);

public record EvalCaseResult(string Question, bool RetrievalHit, bool AnswerCorrect,
    bool RefusalCorrect, double TopScore, string Answer);

public class EvalService(RagService rag)
{
    public async Task<object> RunAsync(IReadOnlyList<EvalCase> cases, CancellationToken ct = default)
    {
        var results = new List<EvalCaseResult>();
        int totalIn = 0, totalOut = 0;
        decimal totalCost = 0;

        foreach (var c in cases)
        {
            var r = await rag.AskAsync(c.Question, ct);
            bool refused = r.Answer.Contains("couldn't find", StringComparison.OrdinalIgnoreCase);

            // Retrieval: was the expected page in the sources?
            bool retrievalHit = !c.Answerable || c.ExpectedPage is null
                || r.Sources.Any(s => s.PageNumber == c.ExpectedPage);

            // Generation: does the answer contain every expected fact?
            bool answerCorrect = !c.Answerable || (!refused
                && (c.ExpectedContains ?? Array.Empty<string>())
                    .All(e => r.Answer.Contains(e, StringComparison.OrdinalIgnoreCase)));

            // Guardrail: refuse only when the question is unanswerable
            bool refusalCorrect = refused == !c.Answerable;

            results.Add(new EvalCaseResult(c.Question, retrievalHit, answerCorrect, refusalCorrect,
                r.Sources.FirstOrDefault()?.Score ?? 0, r.Answer));

            totalIn += r.InputTokens;
            totalOut += r.OutputTokens;
            totalCost += r.EstimatedCost;

            await Task.Delay(1000, ct); // stay under free-tier rate limits
        }

        var answerable = results.Where((_, i) => cases[i].Answerable).ToList();
        static double Pct(int n, int d) => d == 0 ? 0 : Math.Round(100.0 * n / d, 1);

        return new
        {
            summary = new
            {
                totalCases = cases.Count,
                retrievalHitRate = Pct(answerable.Count(r => r.RetrievalHit), answerable.Count),
                answerCorrectness = Pct(answerable.Count(r => r.AnswerCorrect), answerable.Count),
                refusalAccuracy = Pct(results.Count(r => r.RefusalCorrect), results.Count),
                totalInputTokens = totalIn,
                totalOutputTokens = totalOut,
                estimatedCostUsd = Math.Round(totalCost, 6)
            },
            cases = results
        };
    }
}
