import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthenticationService } from '../services/authentication.service.js';

@Component({
    selector: 'ss-login',
    templateUrl: 'src/app/login/login.component.html',
    styleUrls: ['src/app/login/login.component.css'],
})

export class LoginComponent {
    logo: string = 'src/media/logo.jpg';
    username: string;
    password: string;
    errstr: string;

    constructor(private _authenticationService: AuthenticationService,
        private _routerService: Router) {
    }

    login() {
        this._authenticationService.login(this.username, this.password)
            .subscribe(
                result => this.handleLoginResult(result),
                error => console.log(error)
            );
    }

    logout() {
        this._authenticationService.logout()
            .subscribe(null, null, () => {
                localStorage.removeItem('currentUser');
        })
    }

    handleLoginResult(result: any) {
        if (result.errno == '0') {
            this.errstr = "Succesful"            
            localStorage.setItem('currentUser', this.username);
        }
        else{
            this.errstr = result.errstr
        }
    }
}