from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Callable, Iterable, Iterator, Tuple, Any


def run_bounded(
    executor: ThreadPoolExecutor,
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    max_inflight: int,
    should_cancel: Callable[[], bool] = lambda: False,
) -> Iterator[Tuple[Any, Any, BaseException | None]]:
    """Yield (item, result, error) in COMPLETION order, never more than max_inflight pending."""
    it = iter(items)
    pending: dict = {}
    exhausted = False
    try:
        while True:
            while not exhausted and not should_cancel() and len(pending) < max_inflight:
                try:
                    item = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending[executor.submit(fn, item)] = item
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                item = pending.pop(fut)
                err = fut.exception()
                yield item, (None if err else fut.result()), err
            if should_cancel():
                exhausted = True  # stop submitting; drain what is already running
    finally:
        for fut in pending:
            fut.cancel()
