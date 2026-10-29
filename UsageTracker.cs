public class UsageTracker(IConfiguration config)
{
    private readonly decimal _inputPerMillion = config.GetValue<decimal>("Pricing:InputPerMillionTokens");
    private readonly decimal _outputPerMillion = config.GetValue<decimal>("Pricing:OutputPerMillionTokens");
    private readonly object _lock = new();
    private long _requests, _inputTokens, _outputTokens;
    private decimal _totalCost;

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

    public object Snapshot()
    {
        lock (_lock)
            return new
            {
                requests = _requests,
                inputTokens = _inputTokens,
                outputTokens = _outputTokens,
                estimatedCostUsd = Math.Round(_totalCost, 6)
            };
    }
}