import { Component } from '@angular/core';

@Component({
    selector: 'ss-login',
    templateUrl: 'src/app/login/login.component.html'
})

export class LoginComponent {
    logo: string = 'src/media/logo.jpg';
    username: string;
    password: string;

    validate() {
        console.log(this.username)
        console.log(this.password)
    }
}