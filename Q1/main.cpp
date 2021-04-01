
/*
	COSC 4P80 - Assignment 2
	Part 1 - Color Organization
	@author Tennyson Demchuk (td16qg@brocku.ca) | St#: 6190532
	@date 03.22.2021
*/

#include "som.h"
#include <glad/glad.h>		// https://glad.dav1d.de/
#include <GLFW/glfw3.h>		// https://www.glfw.org/
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

/*=========================*/
/*** FUNCTION PROTOTYPES ***/
/*=========================*/
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
void window_resize(GLFWwindow* window, int w, int h);

/*=================*/
/*** GLOBAL VARS ***/
/*=================*/
constexpr int INIT_WIDTH = 500;		// initial window dimensions
constexpr int INIT_HEIGHT = 500;
int width, height;
bool doneTraining;
double start;

/*======================*/
/*** HELPER FUNCTIONS ***/
/*======================*/

// initializes GLAD and loads OpenGL function pointers
void initGLAD() {
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		printf("GLAD init failed.\n");
		glfwTerminate();
		exit(0);
	}
}

// genarate window of specified dimensions
GLFWwindow* createWindow(const str& title, unsigned int w, unsigned int h) {
	GLFWwindow* window = glfwCreateWindow(w, h, title.c_str(), nullptr, nullptr);	// create window
	if (window == nullptr) {
		printf("GLFW window creation failed.\n");
		glfwTerminate();
		exit(0);
	}
	width = w; height = h;
	glfwMakeContextCurrent(window);								// set focus																				
	glfwSetFramebufferSizeCallback(window, window_resize);		// bind resize callback
	glfwSetKeyCallback(window, keyboard);
	return window;
}

// get numeric input value from command line of specified type within specified range
// Entered value stored in 'var'
template <typename T> void getNumericCMD(T& var, const T max = 999999999.0, const T min = 0.0) {
	str line;
	T temp;
	do {
		std::getline(std::cin, line);
		if (line.empty()) return;
		try { temp = (T)std::stod(line); }
		catch (...) {
			printf("Input must be a valid number. Please try again: ");
			continue;
		}
		if (temp >= min && temp <= max) break;
		printf("Input must be in range [%f, %f]. Please try again: ", min, max);
	} while (true);
	var = temp;
}

/*=============================*/
/*** GLFW CALLBACK FUNCTIONS ***/
/*=============================*/
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwTerminate();
		exit(0);
	}
	else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		doneTraining = false;
		start = glfwGetTime();
	}
}

// on window resize
void window_resize(GLFWwindow* window, int w, int h) {
	width = w;
	height = h;
	glViewport(0, 0, width, height);
}

/*===================*/
/*** MAIN FUNCTION ***/
/*===================*/
int main(int argc, char* argv[]) {
	// setup data
	constexpr int	UNIT_DIM	= 3;		// unit vector dimension - 3 for RGB color vector
	constexpr int	N			= 200;		// dimension for NxN grid
	int				in_size		= 1000;		// input vector dimension
	float			learnrate	= 0.3f;		// SOM learning rate
	int				max_iter	= 10000;	// maximum epochs for SOM training
	float			radius		= 30;		// initial neighbourhood radius (sigma)
	printf("COSC 4P80 A2 Q1\nSOM COLOR ORGANIZATION\n");
	printf("======================\n\n");
	printf("Network Size: %d x %d\n", N, N);
	printf("Select Input Size [default %d]: ", in_size);
	getNumericCMD<int>(in_size);
	printf("Select Learning Rate [default %.2f]: ", learnrate);
	getNumericCMD<float>(learnrate, 1.0f, 0.00001f);
	printf("Select # Training Iterations (epochs) [default %d]: ", max_iter);
	getNumericCMD<int>(max_iter, 99999999, 1);
	printf("Select Neighbourhood Radius (sigma) [default %.1f]: ", radius);
	getNumericCMD<float>(radius, (N/2), 1);
	som::grid<UNIT_DIM> pxl;				// grid of pixels (SOM neuron lattice)
	som::grid<UNIT_DIM> in;					// random SOM inputs
	srand(time(0));
	som::initGrid<UNIT_DIM>(N * N, pxl);
	som::randomizeGrid<UNIT_DIM>(pxl);
	som::initGrid<UNIT_DIM>(in_size, in);
	som::randomizeGrid<UNIT_DIM>(in);
	som::SOM<UNIT_DIM> som(in, pxl, N, learnrate, max_iter, radius);

	// setup OpenGL context
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_DECORATED, 0);									// uncomment for undecorated window
	GLFWwindow* window = createWindow("COSC 4P80 - A2", INIT_WIDTH, INIT_HEIGHT);
	initGLAD();

	// generate texture
	uint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, N, N, 0, GL_RGB, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	// bind framebuffer - https://stackoverflow.com/a/30498068
	uint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	printf("\nPress [ESC] to quit.\n");
	printf("Press [SPACE] to start training.\n\n");

	// render loop
	int frameSkip = 15;		// update image display every x frames - less draw call latency
	int frame = 0;
	doneTraining = true;
	while (!glfwWindowShouldClose(window)) {
		// execute 1 training iteration
		if (!doneTraining) if (som.train()) {
			doneTraining = true;
			printf("Finished Training in %fs\n", glfwGetTime() - start);
		}

		if (frame == 0) {				// update texture data - https://stackoverflow.com/a/30498068
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RGB, GL_FLOAT, pxl.data());
			glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
			glBlitFramebuffer(0, 0, N, N, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

			glfwSwapBuffers(window);	// update display
			glfwPollEvents();
		} frame = (frame + 1) % frameSkip;
	}
	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &tex);
	glfwTerminate();
	return 0;
}