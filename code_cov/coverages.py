import abc
import json
import logging
import os
import subprocess
from typing import TypedDict

from typing_extensions import Self


class CreateCoverageRequestData(TypedDict):
    classpathPathnames: list[str]
    focalClasspath: str
    focalClassName: str
    testClassName: str
    testMethodName: str


class Coverage(TypedDict):
    coveredLineNumbers: list[int]


class CodeCov(abc.ABC):
    @abc.abstractmethod
    def create_coverage(
        self: Self,
        request_data: CreateCoverageRequestData,
    ) -> Coverage:
        raise NotImplementedError()


class CodeCovCli(CodeCov):
    def __init__(
        self: Self,
        script_file_pathname: str,
        timeout: int | None = None,
    ) -> None:
        self._script_file_pathname = script_file_pathname
        self._timeout = timeout

    def create_coverage(
        self: Self,
        request_data: CreateCoverageRequestData,
    ) -> Coverage:
        input_json_str = json.dumps(request_data)
        args = [self._script_file_pathname]
        logging.debug('{cwd} [{l}] : ({fcn}, {tcn}, {tmn})'.format(
            l=len(input_json_str),
            cwd=os.getcwd(),
            fcn=request_data['focalClassName'],
            tcn=request_data['testClassName'],
            tmn=request_data['testMethodName'],
        ))
        completed_process = (subprocess
            .run(args, timeout=self._timeout, check=True, capture_output=True,
                text=True, input=input_json_str))
        output = completed_process.stdout
        coverage: Coverage = json.loads(output)
        return coverage
