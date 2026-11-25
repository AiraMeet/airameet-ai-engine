"""Build a GitHub Actions summary from JUnit, coverage, and audit reports."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(os.environ.get("CI_ARTIFACTS", "ci-artifacts"))


def junit_totals() -> tuple[int, int, int, int, float, list[str]]:
    tests = failures = errors = skipped = 0
    duration = 0.0
    files: list[str] = []
    for report in ROOT.glob("**/junit*.xml"):
        files.append(report.name)
        root = ET.parse(report).getroot()
        suites = list(root.iter("testsuite")) if root.tag == "testsuites" else [root]
        for suite in suites:
            tests += int(suite.attrib.get("tests", 0))
            failures += int(suite.attrib.get("failures", 0))
            errors += int(suite.attrib.get("errors", 0))
            skipped += int(suite.attrib.get("skipped", 0))
            duration += float(suite.attrib.get("time", 0))
    return tests, failures, errors, skipped, duration, files


def coverage_percent() -> tuple[str, str]:
    report = next(ROOT.glob("**/coverage.xml"), None)
    if report is None:
        return "Unavailable", "Unavailable"
    root = ET.parse(report).getroot()
    line = 100 * float(root.attrib.get("line-rate", 0))
    branch = 100 * float(root.attrib.get("branch-rate", 0))
    return f"{line:.2f}%", f"{branch:.2f}%"


def issue_counts() -> tuple[int | str, int | str, int | str, int | str, int | str]:
    ruff_path = next(ROOT.glob("**/ruff.json"), None)
    ruff: int | str = "Unavailable"
    if ruff_path:
        ruff = len(json.loads(ruff_path.read_text(encoding="utf-8")))
    mypy_path = next(ROOT.glob("**/mypy.txt"), None)
    mypy: int | str = "Unavailable"
    if mypy_path:
        mypy = sum(
            ": error:" in line
            for line in mypy_path.read_text(encoding="utf-8").splitlines()
        )
    format_path = next(ROOT.glob("**/ruff-format.txt"), None)
    formatting: int | str = "Unavailable"
    if format_path:
        formatting = sum(
            "would be reformatted" in line
            for line in format_path.read_text(encoding="utf-8").splitlines()
        )
    bandit_path = next(ROOT.glob("**/bandit.json"), None)
    bandit: int | str = "Unavailable"
    if bandit_path:
        bandit = len(
            json.loads(bandit_path.read_text(encoding="utf-8")).get("results", [])
        )
    audit_path = next(ROOT.glob("**/pip-audit.json"), None)
    audit: int | str = "Unavailable"
    if audit_path:
        audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
        dependencies = audit_data.get("dependencies", [])
        audit = sum(len(item.get("vulns", [])) for item in dependencies)
    return ruff, formatting, mypy, bandit, audit


def compatibility_results() -> list[str]:
    results = []
    for status_file in ROOT.glob("**/compatibility-status.txt"):
        result = status_file.read_text(encoding="utf-8").strip()
        if result:
            results.append(result)
    return sorted(results)


def artifact_links(run_url: str) -> str:
    artifacts = {
        "Coverage HTML, XML, JUnit, and log": "coverage-and-test-reports",
        "Unit JUnit and log": "unit-test-results",
        "Integration JUnit and log": "integration-test-results",
        "Code quality reports": "code-quality-reports",
        "Security reports": "security-reports",
        "Python compatibility reports": "python-3.10-results, python-3.11-results, python-3.12-results",
    }
    return "\n".join(
        f"- [{label}]({run_url}) — download artifact `{name}`"
        for label, name in artifacts.items()
    )


def main() -> None:
    tests, failures, errors, skipped, duration, junit_files = junit_totals()
    passed = max(tests - failures - errors - skipped, 0)
    line, branch = coverage_percent()
    ruff, formatting, mypy, bandit, audit = issue_counts()
    values = {
        "Code Quality": os.environ["QUALITY_RESULT"],
        "Security Audit": os.environ["SECURITY_RESULT"],
        "Unit Tests": os.environ["UNIT_RESULT"],
        "Integration Tests": os.environ["INTEGRATION_RESULT"],
        "Python Compatibility (3.10–3.12)": os.environ["COMPATIBILITY_RESULT"],
        "Coverage Report": os.environ["COVERAGE_RESULT"],
    }
    compatibility = compatibility_results()
    overall = (
        "Passed"
        if all(value == "success" for value in values.values())
        else "Failed or incomplete"
    )
    run_url = os.environ["RUN_URL"]
    summary = [
        "# AiraMeet AI Engine CI report",
        "",
        f"**Overall status:** {overall}",
        "",
        "## Test execution",
        "",
        f"- Total: {tests}",
        f"- Passed: {passed}",
        f"- Failed: {failures + errors}",
        f"- Skipped: {skipped}",
        f"- Aggregate execution duration: {duration:.2f} seconds (across uploaded JUnit runs)",
        f"- JUnit reports collected: {', '.join(junit_files) if junit_files else 'None'}",
        "",
        "## Coverage",
        "",
        f"- Line coverage: {line}",
        f"- Branch coverage: {branch}",
        "- Per-module coverage and missing lines: HTML report artifact",
        "",
        "## Quality and security findings",
        "",
        f"- Ruff lint findings: {ruff}",
        f"- Ruff formatting findings: {formatting}",
        f"- MyPy errors: {mypy}",
        f"- Ruff formatting and overall quality job: {values['Code Quality']}",
        f"- Bandit findings: {bandit}",
        f"- pip-audit vulnerabilities: {audit}",
        "",
        "## Job results",
        "",
        *[f"- {name}: **{result}**" for name, result in values.items()],
        *[f"- Python {result}" for result in compatibility],
        "",
        "## Reports and artifacts",
        "",
        artifact_links(run_url),
        "",
        "The artifact links open this workflow run; use its **Artifacts** section to download each named report bundle.",
    ]
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
