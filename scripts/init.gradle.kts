allprojects {
    tasks.register("findProjectDir") {
        println(projectDir)
    }
    tasks.register("listSubprojectPaths") {
        for (subproject in project.subprojects) {
            println(subproject.path)
        }
    }
    tasks.register("buildMainRuntimeClasspath") {
        println(project.the<SourceSetContainer>()["main"].runtimeClasspath.asPath)
    }
    tasks.register("buildTestRuntimeClasspath") {
        println(project.the<SourceSetContainer>()["test"].runtimeClasspath.asPath)
    }
}
