import json
import os
import pathlib
import subprocess
import unittest
from unittest import mock

from code_cov import coverages


class CodeCovTest(unittest.TestCase):
    def setUp(self) -> None:
        repo_dir_pathname = os.path.join(os.getcwd(), 'integration_tests',
            'resources', 'repositories', 'maven', 'guess-the-number')
        self._focal_classpath = (
            os.path.join(repo_dir_pathname, 'target', 'classes', ''))
        test_classpath = (
            os.path.join(repo_dir_pathname, 'target', 'test-classes', ''))
        home_dir_pathname = str(pathlib.Path.home())
        maven_dir_pathname = (
            os.path.join(home_dir_pathname, '.m2', 'repository'))
        self._classpath_pathnames = [
            self._focal_classpath,
            test_classpath,
            os.path.join(maven_dir_pathname, 'junit', 'junit', '4.11',
                'junit-4.11.jar'),
            os.path.join(maven_dir_pathname, 'org', 'mockito', 'mockito-core',
                '3.12.4', 'mockito-core-3.12.4.jar'),
            os.path.join(maven_dir_pathname, 'org', 'hamcrest', 'hamcrest-core',
                '1.3', 'hamcrest-core-1.3.jar'),
            os.path.join(maven_dir_pathname, 'net', 'bytebuddy', 'byte-buddy',
                '1.11.13', 'byte-buddy-1.11.13.jar'),
            os.path.join(maven_dir_pathname, 'net', 'bytebuddy',
                'byte-buddy-agent', '1.11.13', 'byte-buddy-agent-1.11.13.jar'),
            os.path.join(maven_dir_pathname, 'org', 'objenesis', 'objenesis',
                '3.2', 'objenesis-3.2.jar'),
        ]
        self._focal_class_name = 'com.example.guessthenumber.ui.CommandLineUi'
        self._test_class_name = (
            'com.example.guessthenumber.ui.CommandLineUiTest')
        self._test_method_name = 'testRun_ioExceptionThrown_exitsGracefully'
        self._request_data = coverages.CreateCoverageRequestData(
            classpathPathnames=self._classpath_pathnames,
            focalClasspath=self._focal_classpath,
            focalClassName=self._focal_class_name,
            testClassName=self._test_class_name,
            testMethodName=self._test_method_name,
        )


class CodeCovCliTest(CodeCovTest):
    def setUp(self) -> None:
        super().setUp()
        self._script_file_pathname = (
            os.path.join(os.getcwd(), 'code-cov-cli', 'bin', 'code-cov-cli'))

    def test_create_coverage__typical_case__creates_coverage(self):
        mock_completed_process = mock.MagicMock()
        input_json_str = json.dumps(self._request_data)
        mock_completed_process.stdout = '{"coveredLineNumbers": [1, 2, 3, 5]}'
        expected_coverage = coverages.Coverage(coveredLineNumbers=[1, 2, 3, 5])
        code_cov_cli = (
            coverages.CodeCovCli(self._script_file_pathname, timeout=10))
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock_completed_process
            actual_coverage = code_cov_cli.create_coverage(self._request_data)
        mock_run.assert_called_once_with(
            [self._script_file_pathname],
            timeout=10,
            check=True,
            capture_output=True,
            text=True,
            input=input_json_str,
        )
        self.assertEqual(expected_coverage, actual_coverage)

    def test_create_coverage__dummy_request_data__creates_coverage(self):
        mock_completed_process = mock.MagicMock()
        request_data = {
            "hello": "world",
            "it's": 1,
            "wonderful": {
                "day": "!",
            },
            'focalClassName': None,
            'testClassName': None,
            'testMethodName': None,
        }
        input_json_str = ('{"hello": "world", "it\'s": 1, '
            + '"wonderful": {"day": "!"}, "focalClassName": null, '
            + '"testClassName": null, "testMethodName": null}')
        mock_completed_process.stdout = '{"coveredLineNumbers": [1, 2, 3, 5]}'
        expected_coverage = coverages.Coverage(coveredLineNumbers=[1, 2, 3, 5])
        code_cov_cli = (
            coverages.CodeCovCli(self._script_file_pathname, timeout=10))
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock_completed_process
            actual_coverage = code_cov_cli.create_coverage(request_data)
        mock_run.assert_called_once_with(
            [self._script_file_pathname],
            timeout=10,
            check=True,
            capture_output=True,
            text=True,
            input=input_json_str,
        )
        self.assertEqual(expected_coverage, actual_coverage)

    def test_create_coverage__analyzer_error__raises_called_process_error(self):
        input_json_str = json.dumps(self._request_data)
        code_cov_cli = (
            coverages.CodeCovCli(self._script_file_pathname, timeout=10))
        with mock.patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(2, '')
            with self.assertRaises(subprocess.CalledProcessError):
                code_cov_cli.create_coverage(self._request_data)
        mock_run.assert_called_once_with(
            [self._script_file_pathname],
            timeout=10,
            check=True,
            capture_output=True,
            text=True,
            input=input_json_str,
        )
