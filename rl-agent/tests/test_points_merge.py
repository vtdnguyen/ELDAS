"""``save_points(merge=True)`` — thêm hạt giống KHÔNG được xoá hạt giống đã có.

Bộ test này tồn tại vì một lần mất dữ liệu thật. ``save_points`` có docstring ghi
"Append/write" nhưng mở tệp bằng ``"w"``, nên một lần chạy thêm hạt giống 47–56 vào
``sweep-LOW/points.jsonl`` đã **xoá sạch** hạt giống 42–46. Không báo lỗi, không cảnh
báo, và bảng kết quả sau đó trông hoàn toàn bình thường — chỉ khác là các phương pháp
học được lặng lẽ đứng trên mười hạt giống trong khi đường cơ sở đứng trên mười lăm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.points import PointRecord, load_points, save_points  # noqa: E402


def pt(method: str, seed: int, energy: float = 100.0,
       sla: float = 1.0) -> PointRecord:
    return PointRecord(method=method, scenario="LOW", seed=seed,
                       energy_kwh=energy, sla_cost=sla, extra={"budget_d": 0.1})


@pytest.fixture()
def path(tmp_path: Path) -> Path:
    return tmp_path / "points.jsonl"


class TestMergeKeepsWhatIsAlreadyThere:
    def test_new_seeds_do_not_erase_old_seeds(self, path):
        save_points([pt("cmdp-d0.1", s) for s in (42, 43, 44)], path)
        save_points([pt("cmdp-d0.1", s) for s in (47, 48)], path, merge=True)
        assert sorted(r.seed for r in load_points(path)) == [42, 43, 44, 47, 48]

    def test_rerunning_a_seed_replaces_it_instead_of_duplicating(self, path):
        """Nhân đôi một hàng sẽ tính hạt giống đó hai lần trong MỌI trung bình và
        khoảng tin cậy phía sau — im lặng và không thể phát hiện từ bảng."""
        save_points([pt("cmdp-d0.1", 42, energy=100.0)], path)
        save_points([pt("cmdp-d0.1", 42, energy=999.0)], path, merge=True)
        rows = load_points(path)
        assert len(rows) == 1
        assert rows[0].energy_kwh == 999.0

    def test_same_seed_different_method_are_two_points(self, path):
        """Ngân sách nằm trong tên phương pháp, nên (method, seed) tách đúng hai
        cấu hình chạy trên cùng một dữ liệu tải."""
        save_points([pt("cmdp-d0.1", 42)], path)
        save_points([pt("cmdp-d0.2", 42)], path, merge=True)
        assert {r.method for r in load_points(path)} == {"cmdp-d0.1", "cmdp-d0.2"}

    def test_merge_into_a_missing_file_just_writes(self, path):
        save_points([pt("cmdp-d0.1", 42)], path, merge=True)
        assert len(load_points(path)) == 1


class TestDefaultStaysReplace:
    def test_without_merge_the_file_is_replaced(self, path):
        """Hành vi mặc định giữ nguyên: một số nơi gọi đúng là muốn ghi đè, và đổi
        ngầm hành vi của chúng sẽ giữ lại những hàng cũ đáng ra phải biến mất."""
        save_points([pt("a", 42), pt("a", 43)], path)
        save_points([pt("b", 47)], path)
        rows = load_points(path)
        assert len(rows) == 1 and rows[0].method == "b"
