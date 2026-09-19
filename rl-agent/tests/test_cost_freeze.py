"""Đóng băng thang chuẩn hoá của tín hiệu ràng buộc.

Vì sao cần: chia chi phí cho độ lệch chuẩn ước lượng từ chính dòng chi phí đó khiến
``J = mean(c)/std(c)`` đo *hình dạng* phân bố chứ không đo *mức*. Chính sách giảm chi phí
thô thì độ lệch chuẩn cũng co theo, nên J gần như đứng yên và ngân sách ``d`` không còn
chỗ bám. Đo trên một đợt chạy đầy đủ: ở LOW, đẩy λ từ 0 lên 3,61 làm chi phí thô giảm
10,6% trong khi J *tăng* 2,2%.

Các test dưới đây khoá tính chất cần có sau khi đóng băng: J tỉ lệ thuận với mức chi phí
thô, và đường cũ (không đóng băng) vẫn nguyên vẹn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reward import RunningScalarNormalizer  # noqa: E402


class TestFreeze:
    def test_khong_dat_thi_van_chay_nhu_cu(self):
        """Mặc định phải giữ nguyên hành vi cũ, không đóng băng."""
        n = RunningScalarNormalizer(center=False)
        for x in (1.0, 5.0, 9.0, 20.0):
            n.update(x)
        assert not n.frozen
        s1 = n.std
        n.update(100.0)
        assert n.std != s1, "thang phải tiếp tục bám theo dòng dữ liệu"

    def test_dong_bang_dung_moc(self):
        n = RunningScalarNormalizer(center=False, freeze_after=4)
        for x in (1.0, 5.0, 9.0, 20.0):
            n.update(x)
        assert n.frozen
        s = n.std
        for x in (1000.0, 2000.0, 3000.0):
            n.update(x)
        assert n.std == s, "sau khi đóng băng, thang không được đổi nữa"

    def test_sau_khi_dong_bang_J_ti_le_voi_muc_chi_phi(self):
        """Tính chất quyết định: J phải phản ánh MỨC, không phải hình dạng.

        Hai dòng chi phí cùng hình dạng nhưng mức lệch nhau 2 lần. Với thang chạy,
        cả hai cho J gần bằng nhau (đúng lỗi cần sửa). Với thang đóng băng chung,
        J của dòng nhẹ hơn phải bằng đúng một nửa.
        """
        mau = [1.0, 2.0, 3.0, 4.0, 5.0] * 6

        def J(nor, dong):
            return sum(nor.update_and_normalize(x) for x in dong) / len(dong)

        # Thang chạy: mỗi dòng tự chuẩn hoá theo chính nó.
        j_cao = J(RunningScalarNormalizer(center=False), [x * 2 for x in mau])
        j_thap = J(RunningScalarNormalizer(center=False), mau)
        assert j_cao == pytest.approx(j_thap, rel=0.05), (
            "đây chính là vấn đề: giảm một nửa chi phí mà J gần như không đổi")

        # Thang đóng băng: cùng một thang cho cả hai dòng.
        warm = [x * 2 for x in mau]
        n_cao = RunningScalarNormalizer(center=False, freeze_after=len(warm))
        for x in warm:
            n_cao.update(x)
        n_thap = RunningScalarNormalizer(center=False, freeze_after=len(warm))
        for x in warm:
            n_thap.update(x)

        j_cao_f = sum(n_cao.normalize(x) for x in warm) / len(warm)
        j_thap_f = sum(n_thap.normalize(x) for x in mau) / len(mau)
        assert j_thap_f == pytest.approx(j_cao_f / 2, rel=1e-9), (
            "sau khi đóng băng, giảm một nửa chi phí phải làm J giảm một nửa")

    def test_van_khong_am(self):
        """Chi phí đã chuẩn hoá phải ≥ 0 để λ·c là một khoản phạt thật."""
        n = RunningScalarNormalizer(center=False, freeze_after=3)
        for x in (2.0, 4.0, 6.0):
            n.update(x)
        assert all(n.normalize(x) >= 0 for x in (0.0, 1.0, 50.0))

    def test_dong_bang_som_van_cho_thang_huu_han(self):
        """Đóng băng ở mẫu thứ nhất thì std chưa xác định; không được ra 0 hay vô cùng."""
        n = RunningScalarNormalizer(center=False, freeze_after=1)
        n.update(7.0)
        assert n.frozen
        assert 0 < n.std < float("inf")
