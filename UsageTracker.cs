/// <summary>
/// Running totals of requests, tokens and estimated cost (FinOps).
/// Prices come from config, so on a free tier this estimates what production would cost.
/// Singleton, so updates are locked.
/// </summary>
public class UsageTracker(IConfiguration config)
{
    private readonly decimal _inputPerMillion = config.GetValue<decimal>("Pricing:InputPerMillionTokens");
    private readonly decimal _outputPerMillion = config.GetValue<decimal>("Pricing:OutputPerMillionTokens");
    private readonly object _lock = new();

    private long _requests, _inputTokens, _outputTokens, _skippedLlmCalls;
    private decimal _totalCost;

    /// <summary>Records one LLM call and returns its estimated cost in USD.</summary>
    public decimal Record(int inputTokens, int outputTokens)
    {
        var cost = inputTokens * _inputPerMillion / 1_000_000m
                 + outputTokens * _outputPerMillion / 1_000_000m;

        lock (_lock)
        {
            _requests++;
            _inputTokens += inputTokens;
            _outputTokens += outputTokens;
            _totalCost += cost;
        }
        return cost;
    }

    /// <summary>Counts questions answered without calling the LLM (below the similarity threshold).</summary>
    public void RecordSkip()
    {
        lock (_lock) _skippedLlmCalls++;
    }

    public object Snapshot()
    {
        lock (_lock)
        {
            return new
            {
                llmRequests = _requests,
                skippedLlmCalls = _skippedLlmCalls,
                inputTokens = _inputTokens,
                outputTokens = _outputTokens,
                estimatedCostUsd = Math.Round(_totalCost, 6),
                averageCostPerRequestUsd = _requests == 0 ? 0 : Math.Round(_totalCost / _requests, 6)
            };
        }
    }
}
