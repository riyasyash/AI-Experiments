from langfuse import get_client

LANGFUSE = get_client()
assert (
    LANGFUSE.auth_check()
), "Langfuse auth failed. Check your keys, and that the host matches your region (EU: https://cloud.langfuse.com, US: https://us.cloud.langfuse.com)."


def log_run(variant_name, agg_metrics):
    with LANGFUSE.start_as_current_observation(
        as_type="span", name=f"rag-eval:{variant_name}"
    ) as span:
        span.update(input={"variant": variant_name}, output=agg_metrics)
        for metric_name, value in agg_metrics.items():
            span.score_trace(name=metric_name, value=float(value), data_type="NUMERIC")
        return span.trace_id
