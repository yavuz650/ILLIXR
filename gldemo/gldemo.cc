#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "common/component.hh"
#include "common/switchboard.hh"
#include "common/data_format.hh"
#include "common/shader_util.hh"
#include "utils/algebra.hh"
#include "block_i.hh"
#include "demo_model.hh"
#include "shaders/blocki_shader.hh"
#include <cmath>

using namespace ILLIXR;

static constexpr int   EYE_TEXTURE_WIDTH   = 1024;
static constexpr int   EYE_TEXTURE_HEIGHT  = 1024;

class gldemo : public component {
public:
	// Public constructor, create_component passes Switchboard handles ("plugs")
	// to this constructor. In turn, the constructor fills in the private
	// references to the switchboard plugs, so the component can read the
	// data whenever it needs to.
	gldemo(std::unique_ptr<writer<rendered_frame>>&& frame_plug,
		  std::unique_ptr<reader_latest<pose_sample>>&& pose_plug,
		  std::unique_ptr<reader_latest<global_config>>&& config_plug)
		: _m_eyebuffer{std::move(frame_plug)}
		, _m_pose{std::move(pose_plug)}
		, _m_config{std::move(config_plug)}
	{ }


	void draw_scene() {

		// OBJ exporter is having winding order issues currently.
		// Please excuse the strange GL_CW and GL_CCW mode switches.

		glFrontFace(GL_CW);

		glBindBuffer(GL_ARRAY_BUFFER, ground_vbo);
		glVertexAttribPointer(vertexPosAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexPosAttr);
		glBindBuffer(GL_ARRAY_BUFFER, ground_normal_vbo);
		glVertexAttribPointer(vertexNormalAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexNormalAttr);
		glUniform4fv(colorUniform, 1, &(ground_color[0]));
		glDrawArrays(GL_TRIANGLES, 0, Ground_plane_NUM_TRIANGLES * 3);

		glFrontFace(GL_CCW);

		glBindBuffer(GL_ARRAY_BUFFER, water_vbo);
		glVertexAttribPointer(vertexPosAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexPosAttr);
		glBindBuffer(GL_ARRAY_BUFFER, water_normal_vbo);
		glVertexAttribPointer(vertexNormalAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexNormalAttr);
		glUniform4fv(colorUniform, 1, &(water_color[0]));
		glDrawArrays(GL_TRIANGLES, 0, Water_plane001_NUM_TRIANGLES * 3);

		glBindBuffer(GL_ARRAY_BUFFER, trees_vbo);
		glVertexAttribPointer(vertexPosAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexPosAttr);
		glBindBuffer(GL_ARRAY_BUFFER, trees_normal_vbo);
		glVertexAttribPointer(vertexNormalAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexNormalAttr);
		glUniform4fv(colorUniform, 1, &(tree_color[0]));
		glDrawArrays(GL_TRIANGLES, 0, Trees_cone_NUM_TRIANGLES * 3);

		glBindBuffer(GL_ARRAY_BUFFER, rocks_vbo);
		glVertexAttribPointer(vertexPosAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexPosAttr);
		glBindBuffer(GL_ARRAY_BUFFER, rocks_normal_vbo);
		glVertexAttribPointer(vertexNormalAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(vertexNormalAttr);
		glUniform4fv(colorUniform, 1, &(rock_color[0]));
		glDrawArrays(GL_TRIANGLES, 0, Rocks_plane002_NUM_TRIANGLES * 3);

		glFrontFace(GL_CCW);
	}

	void main_loop() {
		double lastTime = glfwGetTime();
		glfwMakeContextCurrent(hidden_window);
		while (!_m_terminate.load()) {
			using namespace std::chrono_literals;
			// This "app" is "very slow"!
			//std::this_thread::sleep_for(cosf(glfwGetTime()) * 50ms + 100ms);
			std::this_thread::sleep_for(100ms);
			glUseProgram(demoShaderProgram);

			glBindFramebuffer(GL_FRAMEBUFFER, eyeTextureFBO);

			// Determine which set of eye textures to be using.
			int buffer_to_use = which_buffer.load();

			const pose_sample* pose_ptr = _m_pose->get_latest_ro();

			// We'll calculate this model view matrix
			// using fresh pose data, if we have any.
			ksAlgebra::ksMatrix4x4f modelViewMatrix;

			// Model matrix is just a spinny fun animation
			ksAlgebra::ksMatrix4x4f modelMatrix;
			ksAlgebra::ksMatrix4x4f_CreateRotation(&modelMatrix, 0, 0, 0);

			if(pose_ptr){
				// We have a valid pose from our Switchboard plug.
				pose_t fresh_pose = pose_ptr->pose;
				auto latest_quat = ksAlgebra::ksQuatf {
					.x = fresh_pose.orientation.x,
					.y = fresh_pose.orientation.y,
					.z = fresh_pose.orientation.z,
					.w = fresh_pose.orientation.w
				};
				auto latest_position = ksAlgebra::ksVector3f {
					.x = fresh_pose.position.x,
					.y = fresh_pose.position.y,
					.z = fresh_pose.position.z
				};
				auto scale = ksAlgebra::ksVector3f{1,1,1};
				ksAlgebra::ksMatrix4x4f head_matrix;
				std::cout<< "App using position: " << latest_position.z << std::endl;
				ksAlgebra::ksMatrix4x4f_CreateTranslationRotationScale(&head_matrix, &latest_position, &latest_quat, &scale);
				ksAlgebra::ksMatrix4x4f viewMatrix;
				// View matrix is the inverse of the camera's position/rotation/etc.
				ksAlgebra::ksMatrix4x4f_Invert(&viewMatrix, &head_matrix);
				ksAlgebra::ksMatrix4x4f_Multiply(&modelViewMatrix, &viewMatrix, &modelMatrix);
			} else {
				// We have no pose data from our pose topic :(
				ksAlgebra::ksMatrix4x4f_CreateIdentity(&modelViewMatrix);
			}

			glUseProgram(demoShaderProgram);
			glViewport(0, 0, EYE_TEXTURE_WIDTH, EYE_TEXTURE_HEIGHT);
			glEnable(GL_CULL_FACE);
			glEnable(GL_DEPTH_TEST);
			glClearDepth(1);

			glUniformMatrix4fv(modelViewAttr, 1, GL_FALSE, (GLfloat*)&(modelViewMatrix.m[0][0]));
			glUniformMatrix4fv(projectionAttr, 1, GL_FALSE, (GLfloat*)&(basicProjection.m[0][0]));

			glBindVertexArray(demo_vao);
			
			// Draw things to left eye.
			glBindTexture(GL_TEXTURE_2D_ARRAY, eyeTextures[buffer_to_use]);
			glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, eyeTextures[buffer_to_use], 0, 0);
			glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
			glClearColor(0.6f, 0.8f, 0.9f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			draw_scene();

			
			//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx_vbo);
			//glDrawElements(GL_TRIANGLES, BLOCKI_NUM_POLYS * 3, GL_UNSIGNED_INT, (void*)0);
			
			
			// Draw things to right eye.
			glBindTexture(GL_TEXTURE_2D_ARRAY, eyeTextures[buffer_to_use]);
			glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, eyeTextures[buffer_to_use], 0, 1);
			glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
			glClearColor(0.6f, 0.8f, 0.9f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			draw_scene();

			/*
			glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
			glVertexAttribPointer(vertexPosAttr, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(vertexPosAttr);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx_vbo);
			glDrawElements(GL_TRIANGLES, BLOCKI_NUM_POLYS * 3, GL_UNSIGNED_INT, (void*)0);
			*/
			
			printf("\033[1;32m[GL DEMO APP]\033[0m Submitting frame to buffer %d, frametime %f, FPS: %f\n", buffer_to_use, (float)(glfwGetTime() - lastTime),  (float)(1.0/(glfwGetTime() - lastTime)));
			lastTime = glfwGetTime();
			glFlush();

			// Publish our submitted frame handle to Switchboard!
			auto frame = new rendered_frame;
			frame->texture_handle = eyeTextures[buffer_to_use];
			auto pose = _m_pose->get_latest_ro();
			frame->render_pose = *pose;
			assert(pose);
			which_buffer.store(buffer_to_use == 1 ? 0 : 1);
			_m_eyebuffer->put(frame);
			
		}
		
	}
private:
	std::thread _m_thread;
	std::atomic<bool> _m_terminate {false};
	
	// Switchboard plug for application eye buffer.
	// We're not "writing" the actual buffer data,
	// we're just atomically writing the handle to the
	// correct eye/framebuffer in the "swapchain".
	std::unique_ptr<writer<rendered_frame>> _m_eyebuffer;

	// Switchboard plug for pose prediction.
	std::unique_ptr<reader_latest<pose_sample>> _m_pose;

	// Switchboard plug for global config data, including GLFW/GPU context handles.
	std::unique_ptr<reader_latest<global_config>> _m_config;

	GLFWwindow* hidden_window;

	// These are two eye textures; however, each eye texture
	// really contains two eyes. The reason we have two of
	// them is for double buffering the Switchboard connection.
	GLuint eyeTextures[2];
	GLuint eyeTextureFBO;
	GLuint eyeTextureDepthTarget;

	// This doesn't really need to be atomic right now,
	// as it's only used by the "app's" thread, but 
	// we'll keep it atomic just in case for now!
	std::atomic<int> which_buffer = 0;


	GLuint demo_vao;
	GLuint demoShaderProgram;

	GLuint vertexPosAttr;
	GLuint vertexNormalAttr;
	GLuint modelViewAttr;
	GLuint projectionAttr;

	GLuint ground_vbo;
	GLuint ground_normal_vbo;
	GLuint water_vbo;
	GLuint water_normal_vbo;
	GLuint trees_vbo;
	GLuint trees_normal_vbo;
	GLuint rocks_vbo;
	GLuint rocks_normal_vbo;

	GLuint colorUniform;

	GLfloat water_color[4] = {
		0.0, 0.3, 0.5, 1.0
	};

	GLfloat ground_color[4] = {
		0.1, 0.2, 0.1, 1.0
	};

	GLfloat tree_color[4] = {
		0.0, 0.3, 0.0, 1.0
	};

	GLfloat rock_color[4] = {
		0.3, 0.3, 0.3, 1.0
	};

	ksAlgebra::ksMatrix4x4f basicProjection;


	static void GLAPIENTRY
	MessageCallback( GLenum source,
					GLenum type,
					GLuint id,
					GLenum severity,
					GLsizei length,
					const GLchar* message,
					const void* userParam )
	{
	fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
			( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
				type, severity, message );
	}

	int createSharedEyebuffer(GLuint* texture_handle){

		// Create the shared eye texture handle.
		glGenTextures(1, texture_handle);
		glBindTexture(GL_TEXTURE_2D_ARRAY, *texture_handle);

		// Set the texture parameters for the texture that the FBO will be
		// mapped into.
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, EYE_TEXTURE_WIDTH, EYE_TEXTURE_HEIGHT, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

		glBindTexture(GL_TEXTURE_2D_ARRAY, 0); // unbind texture, will rebind later

		if(glGetError()){
			return 0;
		} else {
			return 1;
		}
	}

	void createFBO(GLuint* texture_handle, GLuint* fbo, GLuint* depth_target){
		// Create a framebuffer to draw some things to the eye texture
		glGenFramebuffers(1, fbo);
		// Bind the FBO as the active framebuffer.
    	glBindFramebuffer(GL_FRAMEBUFFER, *fbo);

		glGenRenderbuffers(1, depth_target);
    	glBindRenderbuffer(GL_RENDERBUFFER, *depth_target);
    	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, EYE_TEXTURE_WIDTH, EYE_TEXTURE_HEIGHT);
    	//glRenderbufferStorageMultisample(GL_RENDERBUFFER, fboSampleCount, GL_DEPTH_COMPONENT, EYE_TEXTURE_WIDTH, EYE_TEXTURE_HEIGHT);
    	glBindRenderbuffer(GL_RENDERBUFFER, 0);

		// Bind eyebuffer texture
		printf("About to bind eyebuffer texture, texture handle: %d\n", *texture_handle);
		glBindTexture(GL_TEXTURE_2D_ARRAY, *texture_handle);
		glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, *texture_handle, 0, 0);
    	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

		// attach a renderbuffer to depth attachment point
    	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, *depth_target);

		if(glGetError()){
        	printf("displayCB, error after creating fbo\n");
    	}

		// Unbind FBO.
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

public:
	/* compatibility interface */

	// Dummy "application" overrides _p_start to control its own lifecycle/scheduling.
	// This may be changed later, but it really doesn't matter for this purpose because
	// it will be replaced by a real, Monado-interfaced application.
	virtual void _p_start() override {
		// Create a hidden window, as we're drawing the demo "offscreen"
		glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

		auto fetched_config = _m_config->get_latest_ro();
		if(!fetched_config){
			std::cerr << "Dummy GLDEMO app failed to fetch global config." << std::endl;
		}
		hidden_window = glfwCreateWindow(1024, 1024, "GL Demo App", 0, fetched_config->glfw_context);

		if(hidden_window == NULL){
			printf("Whoa, what?");
		}

		glfwMakeContextCurrent(hidden_window);

		glEnable              ( GL_DEBUG_OUTPUT );
		glDebugMessageCallback( MessageCallback, 0 );
		
		// Init and verify GLEW
		if(glewInit()){
			printf("Failed to init GLEW\n");
			glfwDestroyWindow(hidden_window);
		}

		// Create two shared eye textures.
		// Note; each "eye texture" actually contains two eyes.
		// The two eye textures here are actually for double-buffering
		// the Switchboard connection.
		createSharedEyebuffer(&(eyeTextures[0]));
		createSharedEyebuffer(&(eyeTextures[1]));

		// Initialize FBO and depth targets, attaching to the frame handle
		createFBO(&(eyeTextures[0]), &eyeTextureFBO, &eyeTextureDepthTarget);

		// Create and bind global VAO object
		glGenVertexArrays(1, &demo_vao);
    	glBindVertexArray(demo_vao);

		demoShaderProgram = init_and_link(blocki_vertex_shader, blocki_fragment_shader);
		std::cout << "Demo app shader program is program " << demoShaderProgram << std::endl;

		vertexPosAttr = glGetAttribLocation(demoShaderProgram, "vertexPosition");
		vertexNormalAttr = glGetAttribLocation(demoShaderProgram, "vertexNormal");
		modelViewAttr = glGetUniformLocation(demoShaderProgram, "u_modelview");
		projectionAttr = glGetUniformLocation(demoShaderProgram, "u_projection");

		colorUniform = glGetUniformLocation(demoShaderProgram, "u_color");

		// Config mesh position vbo
		glGenBuffers(1, &ground_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, ground_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Ground_plane_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Ground_Plane_vertex_data[0]), GL_STATIC_DRAW);
		glGenBuffers(1, &water_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, water_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Water_plane001_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Water_Plane001_vertex_data[0]), GL_STATIC_DRAW);
		glGenBuffers(1, &trees_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, trees_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Trees_cone_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Trees_Cone_vertex_data[0]), GL_STATIC_DRAW);
		glGenBuffers(1, &rocks_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, rocks_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Rocks_plane002_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Rocks_Plane002_vertex_data[0]), GL_STATIC_DRAW);

		glGenBuffers(1, &ground_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, ground_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Ground_plane_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Ground_Plane_normal_data[0]), GL_STATIC_DRAW);
		glGenBuffers(1, &water_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, water_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Water_plane001_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Water_Plane001_normal_data[0]), GL_STATIC_DRAW);
		glGenBuffers(1, &trees_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, trees_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Trees_cone_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Trees_Cone_normal_data[0]), GL_STATIC_DRAW);
		glGenBuffers(1, &rocks_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, rocks_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, (Rocks_plane002_NUM_TRIANGLES * 3 * 3) * sizeof(GLfloat), &(Rocks_Plane002_normal_data[0]), GL_STATIC_DRAW);
		
		glVertexAttribPointer(vertexPosAttr, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glVertexAttribPointer(vertexNormalAttr, 3, GL_FLOAT, GL_FALSE, 0, 0);

		// Config mesh indices vbo
		//glGenBuffers(1, &idx_vbo);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx_vbo);
		//glBufferData(GL_ELEMENT_ARRAY_BUFFER, (BLOCKI_NUM_POLYS * 3) * sizeof(GLuint), logo3d_poly_data, GL_STATIC_DRAW);

		// Construct a basic perspective projection
		ksAlgebra::ksMatrix4x4f_CreateProjectionFov( &basicProjection, 40.0f, 40.0f, 40.0f, 40.0f, 0.03f, 20.0f );

		glfwMakeContextCurrent(NULL);

		_m_thread = std::thread{&gldemo::main_loop, this};

	}

	virtual void _p_stop() override {
		_m_terminate.store(true);
		_m_thread.join();
	}

	virtual ~gldemo() override {
		// TODO: need to cleanup here!
	}
};

extern "C" component* create_component(switchboard* sb) {
	/* First, we declare intent to read/write topics. Switchboard
	   returns handles to those topics. */
	
	// We publish application frames to Switchboard.
	auto frame_ev = sb->publish<rendered_frame>("eyebuffer");

	// We sample the up-to-date, predicted pose.
	auto pose_ev = sb->subscribe_latest<pose_sample>("pose");

	// We need global config data to create a shared GLFW context.
	auto config_ev = sb->subscribe_latest<global_config>("global_config");

	return new gldemo {std::move(frame_ev), std::move(pose_ev), std::move(config_ev)};
}

