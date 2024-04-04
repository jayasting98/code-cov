import abc
import glob
import logging
import os
import subprocess
from xml.etree import ElementTree

from typing_extensions import Self

from code_cov import utilities


class Project(abc.ABC):
    @abc.abstractmethod
    def find_subproject_pathnames(self: Self) -> list[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self: Self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def find_classpath_pathnames(self: Self) -> list[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def find_focal_classpath(self: Self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def root_dir_pathname(self: Self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def project_dir_pathname(self: Self) -> str:
        raise NotImplementedError()


class MavenProject(Project):
    def __init__(
        self: Self,
        root_dir_pathname: str,
        project_dir_pathname: str,
    ) -> None:
        self._root_dir_pathname = root_dir_pathname
        self._project_dir_pathname = project_dir_pathname

    def find_subproject_pathnames(self: Self) -> list[str]:
        pathnames = [self._root_dir_pathname]
        subproject_pathnames = list()
        i = 0
        while i < len(pathnames):
            pathname = pathnames[i]
            i += 1
            logging.info(f'finding subprojects: {pathname}')
            subproject_pathnames.append(pathname)
            args = ['mvn', 'help:evaluate', '-Dexpression=project.modules',
                '-q', '-DforceStdout']
            with utilities.WorkingDirectory(pathname):
                completed_process = (
                    subprocess.run(args, capture_output=True, text=True))
            if completed_process.returncode != 0:
                logging.warn(f'could not find subprojects: {pathname}')
                continue
            modules_xml_str = completed_process.stdout
            modules_element = ElementTree.fromstring(modules_xml_str)
            module_elements = modules_element.findall('string')
            if len(module_elements) < 1:
                continue
            module_names = [
                module_element.text for module_element in module_elements]
            child_pathnames = [os.path.join(pathname, module_name)
                for module_name in module_names]
            pathnames.extend(child_pathnames)
        return subproject_pathnames

    def compile(self: Self) -> None:
        args = ['mvn', 'test-compile']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = subprocess.run(args, stdout=subprocess.DEVNULL)
        completed_process.check_returncode()

    def find_classpath_pathnames(self: Self) -> list[str]:
        args = ['mvn', 'dependency:build-classpath',
            '-Dmdep.outputFile=/dev/stdout', '-q']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        output = completed_process.stdout
        classpath_pathnames = output.split(os.pathsep)
        classpath_pathnames.append(self.find_focal_classpath())
        classpath_pathnames.append(self._find_test_classpath())
        classpath_pathnames.append(self._find_focal_resources_classpath())
        classpath_pathnames.append(self._find_test_resources_classpath())
        return classpath_pathnames

    def find_focal_classpath(self: Self) -> str:
        args = ['mvn', 'help:evaluate',
            '-Dexpression=project.build.outputDirectory', '-q', '-DforceStdout']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        focal_classpath = completed_process.stdout + os.path.sep
        return focal_classpath

    @property
    def root_dir_pathname(self: Self) -> str:
        return self._root_dir_pathname

    @property
    def project_dir_pathname(self: Self) -> str:
        return self._project_dir_pathname

    def _find_test_classpath(self: Self) -> str:
        args = ['mvn', 'help:evaluate',
            '-Dexpression=project.build.testOutputDirectory', '-q',
            '-DforceStdout']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        test_classpath = completed_process.stdout + os.path.sep
        return test_classpath

    def _find_focal_resources_classpath(self: Self) -> str:
        args = ['mvn', 'help:evaluate',
            '-Dexpression=project.build.resources[0].directory', '-q',
            '-DforceStdout']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        focal_resources_classpath = completed_process.stdout + os.path.sep
        return focal_resources_classpath

    def _find_test_resources_classpath(self: Self) -> str:
        args = ['mvn', 'help:evaluate',
            '-Dexpression=project.build.testResources[0].directory',
            '-q', '-DforceStdout']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        test_resources_classpath = completed_process.stdout + os.path.sep
        return test_resources_classpath


_gradle_init_script_pathname = (
    os.path.join(os.getcwd(), 'scripts', 'init.gradle.kts'))


class GradleProject(Project):
    def __init__(
        self: Self,
        root_dir_pathname: str,
        project_dir_pathname: str,
    ) -> None:
        self._root_dir_pathname = root_dir_pathname
        self._project_dir_pathname = project_dir_pathname
        self._init_script_rel_pathname = (os.path
            .relpath(_gradle_init_script_pathname, self._project_dir_pathname))

    def find_subproject_pathnames(self: Self) -> list[str]:
        pathnames = [self._root_dir_pathname]
        paths = ['']
        unique_paths = {''}
        subproject_pathnames = list()
        i = 0
        while i < len(pathnames):
            pathname = pathnames[i]
            path = paths[i]
            i += 1
            logging.info(f'finding subprojects: {pathname}')
            subproject_pathnames.append(pathname)
            init_script_rel_pathname = (os.path
                .relpath(_gradle_init_script_pathname, pathname))
            args = [self.gradle_command, '-q', '--init-script',
                init_script_rel_pathname, f'{path}:listSubprojectPaths']
            with utilities.WorkingDirectory(pathname):
                completed_process = (
                    subprocess.run(args, capture_output=True, text=True))
            if completed_process.returncode != 0:
                logging.warn(f'could not find subprojects: {pathname}')
                continue
            project_paths_str = completed_process.stdout
            child_paths = project_paths_str.strip().split(os.linesep)
            for child_path in child_paths:
                if child_path in unique_paths:
                    continue
                unique_paths.add(child_path)
                child_rel_path = child_path[1:]
                child_rel_pathname = child_rel_path.replace(':', os.path.sep)
                child_pathname = (
                    os.path.join(self._root_dir_pathname, child_rel_pathname))
                pathnames.append(child_pathname)
                paths.append(child_path)
        return subproject_pathnames

    def compile(self: Self) -> None:
        project_name = self._find_project_name()
        args = [self.gradle_command, f'{project_name}:testClasses']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = subprocess.run(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        completed_process.check_returncode()

    def find_classpath_pathnames(self: Self) -> list[str]:
        project_name = self._find_project_name()
        args = [self.gradle_command, '-q', '--init-script',
            self._init_script_rel_pathname,
            f'{project_name}:buildTestRuntimeClasspath']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        output = completed_process.stdout
        line = output.split(os.linesep)[0]
        pathnames = line.split(os.pathsep)
        classpath_pathnames = [
            pathname if pathname.endswith('.jar') else pathname + os.path.sep
            for pathname in pathnames]
        return classpath_pathnames

    def find_focal_classpath(self: Self) -> str:
        project_name = self._find_project_name()
        args = [self.gradle_command, '-q', '--init-script',
            self._init_script_rel_pathname,
            f'{project_name}:buildTestRuntimeClasspath']
        with utilities.WorkingDirectory(self._project_dir_pathname):
            completed_process = (
                subprocess.run(args, capture_output=True, text=True))
        completed_process.check_returncode()
        output = completed_process.stdout
        line = output.split(os.linesep)[0]
        classpath_pathnames = line.split(os.pathsep)
        candidates = [pathname for pathname in classpath_pathnames
            if 'main' in pathname and 'classes' in pathname]
        focal_classpath = candidates[0] + os.path.sep
        return focal_classpath

    @property
    def root_dir_pathname(self: Self) -> str:
        return self._root_dir_pathname

    @property
    def project_dir_pathname(self: Self) -> str:
        return self._project_dir_pathname

    def _find_project_name(self: Self) -> str:
        try:
            return self._project_name
        except AttributeError:
            if self._root_dir_pathname == self._project_dir_pathname:
                self._project_name = str()
                return self._project_name
            project_rel_pathname = (os.path
                .relpath(self._project_dir_pathname, self._root_dir_pathname))
            project_name = ':' + project_rel_pathname.replace(os.path.sep, ':')
            self._project_name = project_name
            return self._project_name

    @property
    def gradle_command(self: Self) -> str:
        try:
            return self._gradle_command
        except AttributeError:
            is_windows = os.name == 'nt'
            if is_windows:
                gradlew_pathname = (
                    os.path.join(self._root_dir_pathname, 'gradlew.bat'))
            else:
                gradlew_pathname = (
                    os.path.join(self._root_dir_pathname, 'gradlew'))
            if os.path.isfile(gradlew_pathname):
                os.chmod(gradlew_pathname, 0o777)
                self._gradle_command = gradlew_pathname
            else:
                self._gradle_command = 'gradle'
            return self._gradle_command

def create_project(
    root_dir_pathname: str,
    project_dir_pathname: str,
) -> Project:
    if os.path.isfile(os.path.join(project_dir_pathname, 'pom.xml')):
        return MavenProject(root_dir_pathname, project_dir_pathname)
    if (os.path.isfile(os.path.join(project_dir_pathname, 'build.gradle'))
        or (os.path
            .isfile(os.path.join(project_dir_pathname, 'build.gradle.kts')))):
        return GradleProject(root_dir_pathname, project_dir_pathname)
    possible_build_gradle_pathnames = (glob
        .glob('**/build.gradle', recursive=True, root_dir=root_dir_pathname))
    possible_build_gradle_pathnames.extend(glob.glob(
        '**/build.gradle.kts', recursive=True, root_dir=root_dir_pathname))
    for file_pathname in possible_build_gradle_pathnames:
        file_full_pathname = os.path.join(root_dir_pathname, file_pathname)
        if os.path.isfile(file_full_pathname):
            return GradleProject(root_dir_pathname, project_dir_pathname)
    raise ValueError()
