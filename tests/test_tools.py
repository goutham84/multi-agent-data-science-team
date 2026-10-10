from ds_team import tools


def test_paths_outside_the_project_are_refused(project):
    assert tools.list_files.invoke({"directory": ".."}).startswith("Error: PermissionError")
    assert tools.read_file.invoke({"path": "../secret.txt"}).startswith("Error: PermissionError")


def test_list_csv_files(project):
    assert tools.list_csv_files.invoke({}).startswith("data/sample.csv")


def test_read_csv_file_never_returns_more_than_30_rows(project):
    result = tools.read_csv_file.invoke({"path": "data/sample.csv", "rows": 100})
    header, table = result.split("\n\n", 1)
    assert header == "data/sample.csv: 20 rows x 8 columns"
    assert len(table.splitlines()) == 1 + 20  # column names + every row of the small file

    result = tools.read_csv_file.invoke({"path": "data/sample.csv", "rows": 5})
    assert len(result.split("\n\n", 1)[1].splitlines()) == 1 + 5


def test_dataframe_operations(project):
    def run(**args):
        return tools.dataframe_operation.invoke({"path": "data/sample.csv", **args})

    assert run(operation="shape") == "20 rows x 8 columns"
    assert "numerical: ['year', 'selling_price', 'km_driven']" in run(operation="columns")
    assert "Petrol" in run(operation="value_counts", column="fuel")
    assert run(operation="value_counts").startswith("Error: ValueError")
    assert "not found" in run(operation="describe", column="price")


def test_create_chart_saves_png(project):
    result = tools.create_chart.invoke(
        {"path": "data/sample.csv", "chart_type": "histogram", "x": "selling_price"}
    )
    assert result == "Chart saved to plots/histogram_selling_price.png"
    assert (project / "plots" / "histogram_selling_price.png").exists()

    result = tools.create_chart.invoke({"path": "data/sample.csv", "chart_type": "scatter", "x": "year"})
    assert result.startswith("Error: ValueError")


def test_run_python_file(project):
    (project / "hello.py").write_text("print('hello from the agent')")
    assert tools.run_python_file.invoke({"path": "hello.py"}) == "exit code: 0\nhello from the agent"
    assert tools.run_python_file.invoke({"path": "data/sample.csv"}).startswith("Error: ValueError")


def test_shell_only_runs_allow_listed_commands(project):
    for command in ["rm -rf .", "uv run python -c 1", "uv run ../outside.py", "git push"]:
        assert tools.run_shell_command.invoke({"command": command}).startswith("Error: PermissionError")


def test_install_package_rejects_option_injection(project):
    assert tools.install_package.invoke({"package": "--index-url=http://evil"}).startswith("Error: ValueError")
