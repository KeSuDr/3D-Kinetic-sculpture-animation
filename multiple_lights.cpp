#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>

#include <iostream>
#include <vector>
#include <cmath>

#define M_PI 3.14159265358979323846

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// lighting
glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

// GLOBAL VARIABLES FOR SPAWNING (NEW)
std::vector<glm::vec3> spawnedSuperellipsoids;
bool e_pressed_last_frame = false;

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

void generateSuperellipsoid(
    std::vector<Vertex>& vertices,
    std::vector<unsigned int>& indices,
    float a, float b, float c,
    float n1, float n2,
    int stacks = 64, int slices = 64)
{
    vertices.clear();
    indices.clear();

    auto sgn = [](float x) { return (x < 0) ? -1.0f : 1.0f; };
    auto powe = [&](float base, float exp) {
        return sgn(base) * std::pow(std::abs(base), exp);
        };

    for (int i = 0; i <= stacks; i++) {
        float u = -M_PI / 2.0f + (float)i / stacks * M_PI;
        for (int j = 0; j <= slices; j++) {
            float v = -M_PI + (float)j / slices * 2.0f * M_PI;

            float cu = cos(u), su = sin(u);
            float cv = cos(v), sv = sin(v);

            float x = a * powe(cu, n1) * powe(cv, n2);
            float y = b * powe(cu, n1) * powe(sv, n2);
            float z = c * powe(su, n1);

            glm::vec3 pos(x, y, z);

            // approximate normal
            glm::vec3 n = glm::normalize(glm::vec3(
                x / (a * a), y / (b * b), z / (c * c)
            ));

            glm::vec2 tex(
                (float)j / slices,
                (float)i / stacks
            );

            vertices.push_back({ pos, n, tex });
        }
    }

    for (int i = 0; i < stacks; i++) {
        for (int j = 0; j < slices; j++) {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Superellipsoid Morphing", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader lightingShader("6.multiple_lights.vs", "6.multiple_lights.fs");
    Shader lightCubeShader("6.light_cube.vs", "6.light_cube.fs");

    // ====================================================================
    // 1. SUPER ELLIPSOID MESH SETUP (REPLACES CUBE VERTEX DATA)
    // ====================================================================

    // Mesh storage
    std::vector<Vertex> superellipsoidVertices;
    std::vector<unsigned int> superellipsoidIndices;

    // Generate initial shape (sphere: a=b=c=1, n1=n2=1)
    generateSuperellipsoid(superellipsoidVertices, superellipsoidIndices, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    unsigned int superellipsoidVAO, VBO, EBO;
    glGenVertexArrays(1, &superellipsoidVAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(superellipsoidVAO);

    // Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Use GL_DYNAMIC_DRAW since the geometry will change every frame (for morphing)
    glBufferData(GL_ARRAY_BUFFER, superellipsoidVertices.size() * sizeof(Vertex), superellipsoidVertices.data(), GL_DYNAMIC_DRAW);

    // Element Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, superellipsoidIndices.size() * sizeof(unsigned int), superellipsoidIndices.data(), GL_DYNAMIC_DRAW);

    // Vertex Attributes
    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    // Normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    glEnableVertexAttribArray(1);
    // TexCoords
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    glEnableVertexAttribArray(2);

    // Unbind VAO
    glBindVertexArray(0);

    // ====================================================================
    // 2. LIGHT CUBE SETUP 
    // ====================================================================
    float lightCubeVertices[] = { /* same cube data as your original 'vertices' array, but only positions (3 floats) needed for the light cube shader */
             -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
             -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
             -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
              0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
             -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f,
             -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f
    };

    unsigned int lightCubeVAO, lightCubeVBO;
    glGenVertexArrays(1, &lightCubeVAO);
    glGenBuffers(1, &lightCubeVBO);

    glBindBuffer(GL_ARRAY_BUFFER, lightCubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lightCubeVertices), lightCubeVertices, GL_STATIC_DRAW);

    glBindVertexArray(lightCubeVAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // positions of the point lights
    glm::vec3 pointLightPositions[] = {
        glm::vec3(0.7f,  0.2f,  2.0f),
        glm::vec3(2.3f, -3.3f, -4.0f),
        glm::vec3(-4.0f,  2.0f, -12.0f),
        glm::vec3(0.0f,  0.0f, -3.0f)
    };
    // ====================================================================

    // load textures
    unsigned int diffuseMap = loadTexture(FileSystem::getPath("resources/textures/Solid_yellow.png").c_str());
    //unsigned int specularMap = loadTexture(FileSystem::getPath("resources/textures/wood.png").c_str());

    // shader configuration
    lightingShader.use();
    lightingShader.setInt("material.diffuse", 0);
    lightingShader.setInt("material.specular", 1);

    printf("Press E to summon superellipsoid \n");


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        processInput(window);

        // render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ====================================================================
        // 3. MORPHING LOGIC AND BUFFER UPDATE
        // ====================================================================

        // Calculate morph parameters for superellipsoid
        float t = glfwGetTime();
        float n1 = 0.2f + 1.8f * (std::sin(t * 1.2f) * 0.5f + 0.5f); // 0.2 to 2.0
        float n2 = 0.2f + 1.8f * (std::cos(t * 0.8f) * 0.5f + 0.5f); // 0.2 to 2.0

        // Regenerate and update geometry buffers (Note: All superellipsoids use this shape)
        generateSuperellipsoid(superellipsoidVertices, superellipsoidIndices, 1.0f, 1.0f, 1.0f, n1, n2);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, superellipsoidVertices.size() * sizeof(Vertex), superellipsoidVertices.data());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, superellipsoidIndices.size() * sizeof(unsigned int), superellipsoidIndices.data());

        // ====================================================================

        // Lighting setup
        lightingShader.use();
        lightingShader.setVec3("viewPos", camera.Position);
        lightingShader.setFloat("material.shininess", 32.0f);

        // NEW: Set explicit material colors for a vibrant blue with bright blue reflections
        lightingShader.setVec3("material.ambient", 0.05f, 0.1f, 0.3f);   // Darker blue base
        lightingShader.setVec3("material.diffuse", 0.2f, 0.5f, 0.8f);    // Main body bright blue
        lightingShader.setVec3("material.specular", 0.7f, 0.9f, 1.0f);  // Bright blue/cyan reflections (light reflexes)

        // Set all light uniforms here...
        // directional light
        lightingShader.setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
        lightingShader.setVec3("dirLight.ambient", 0.05f, 0.05f, 0.05f);
        lightingShader.setVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
        lightingShader.setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);
        // point lights
        for (unsigned int i = 0; i < 4; i++)
        {
            std::string name = "pointLights[" + std::to_string(i) + "]";
            lightingShader.setVec3(name + ".position", pointLightPositions[i]);
            lightingShader.setVec3(name + ".ambient", 0.05f, 0.05f, 0.05f);
            lightingShader.setVec3(name + ".diffuse", 0.8f, 0.8f, 0.8f);
            lightingShader.setVec3(name + ".specular", 1.0f, 1.0f, 1.0f);
            lightingShader.setFloat(name + ".constant", 1.0f);
            lightingShader.setFloat(name + ".linear", 0.09f);
            lightingShader.setFloat(name + ".quadratic", 0.032f);
        }
        // spotLight
        lightingShader.setVec3("spotLight.position", camera.Position);
        lightingShader.setVec3("spotLight.direction", camera.Front);
        lightingShader.setVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
        lightingShader.setVec3("spotLight.diffuse", 1.0f, 1.0f, 1.0f);
        lightingShader.setVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
        lightingShader.setFloat("spotLight.constant", 1.0f);
        lightingShader.setFloat("spotLight.linear", 0.09f);
        lightingShader.setFloat("spotLight.quadratic", 0.032f);
        lightingShader.setFloat("spotLight.cutOff", glm::cos(glm::radians(12.5f)));
        lightingShader.setFloat("spotLight.outerCutOff", glm::cos(glm::radians(15.0f)));


        // view/projection transformations
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        lightingShader.setMat4("projection", projection);
        lightingShader.setMat4("view", view);

        // bind textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseMap);
        glActiveTexture(GL_TEXTURE1);
        //glBindTexture(GL_TEXTURE_2D, specularMap);

        // --- RENDER ALL SUPER ELLIPSOIDS ---
        glBindVertexArray(superellipsoidVAO);

        // 1. RENDER THE MORPHING SUPER ELLIPSOID (at world origin)
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, t * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));
        lightingShader.setMat4("model", model);
        glDrawElements(GL_TRIANGLES, superellipsoidIndices.size(), GL_UNSIGNED_INT, 0);

        // 2. RENDER ALL SPAWNED SUPER ELLIPSOIDS (using the same *morphing* shape)
        for (const auto& position : spawnedSuperellipsoids)
        {
            model = glm::mat4(1.0f);
            model = glm::translate(model, position);
            // Add a small rotation and scale to the spawned objects
            model = glm::rotate(model, (float)glfwGetTime() * 0.2f, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(0.5f)); // Make the spawned objects smaller
            lightingShader.setMat4("model", model);
            glDrawElements(GL_TRIANGLES, superellipsoidIndices.size(), GL_UNSIGNED_INT, 0);
        }

        // ====================================================================

        // also draw the lamp object(s)
        lightCubeShader.use();
        lightCubeShader.setMat4("projection", projection);
        lightCubeShader.setMat4("view", view);

        // we now draw as many light bulbs as we have point lights.
        glBindVertexArray(lightCubeVAO);
        for (unsigned int i = 0; i < 4; i++)
        {
            model = glm::mat4(1.0f);
            model = glm::translate(model, pointLightPositions[i]);
            model = glm::scale(model, glm::vec3(0.2f)); // Make it a smaller cube
            lightCubeShader.setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 36); // The light cube has 36 vertices (12 triangles)
        }


        // glfw: swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // de-allocate all resources
    glDeleteVertexArrays(1, &superellipsoidVAO);
    glDeleteVertexArrays(1, &lightCubeVAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &lightCubeVBO);

    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

    // NEW: Spawn logic for 'E' key
    bool e_is_pressed = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;

    if (e_is_pressed && !e_pressed_last_frame)
    {
        // Add the current camera position, pushed forward by 2.0f units
        glm::vec3 spawn_pos = camera.Position + camera.Front * 2.0f;
        spawnedSuperellipsoids.push_back(spawn_pos);
        // Note: You must have <iostream> included for this to work
        std::cout << "Superellipsoid spawned at: (" << spawn_pos.x << ", " << spawn_pos.y << ", " << spawn_pos.z << ")" << std::endl;
    }

    e_pressed_last_frame = e_is_pressed;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

// utility function for loading a 2D texture from file
// ---------------------------------------------------
unsigned int loadTexture(char const* path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}